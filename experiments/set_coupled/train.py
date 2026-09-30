"""Train A1 with training-only set-level source-target retrieval and distillation."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v0.data import (DEFAULT_BIDA_ROOT, DEFAULT_DATA_DIR, source_loaders,
                                 target_loader, target_train_loader)
from experiments.set_coupled.model import SetCoupledA1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distillation_kl(student_logits, teacher_logits):
    teacher = F.softmax(teacher_logits.detach(), dim=1)
    student = F.log_softmax(student_logits, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean")


def metrics(labels, predictions, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    counts = cm.sum(1)
    recalls = np.divide(np.diag(cm), counts, out=np.zeros(classes, dtype=float), where=counts != 0)
    return {"n": int(cm.sum()), "oa_percent": float(np.trace(cm) / cm.sum() * 100),
            "aa_percent": float(recalls.mean() * 100),
            "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
            "per_class_percent": (recalls * 100).tolist(),
            "prediction_counts": cm.sum(0).tolist(), "confusion_matrix": cm.tolist()}


@torch.inference_mode()
def evaluate_self(model, loader, device, classes):
    model.eval()
    labels, predictions = [], []
    for images, truth in loader:
        output = model.self_forward(images.to(device))
        labels.append(truth.numpy())
        predictions.append(output["logits_self"].argmax(1).cpu().numpy())
    return metrics(np.concatenate(labels), np.concatenate(predictions), classes)


def attention_entropy(attention):
    probabilities = attention.clamp_min(1e-12)
    entropy = -(probabilities * probabilities.log()).sum(-1)
    return entropy.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("c0", "c1"), required=True)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-distill", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learnable-alpha", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bida-root", type=Path, default=DEFAULT_BIDA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-target-eval", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lambda_distill < 0:
        parser.error("epochs must be positive and lambda-distill nonnegative")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    source_train, source_val, classes, bands = source_loaders(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size,
        args.num_workers, args.patch_size)
    target_train = target_train_loader(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size,
        args.num_workers, args.patch_size, generator=source_train.generator)
    model = SetCoupledA1(bands=bands, classes=classes, patch_size=args.patch_size,
                         alpha=args.alpha, learnable_alpha=args.learnable_alpha).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    config = {"arm": args.arm, "seed": args.seed, "epochs": args.epochs,
              "batch_size": args.batch_size, "patch_size": args.patch_size,
              "lr": args.lr, "optimizer": "SGD, no momentum or schedule (Strict BiDA)",
              "objective": ("source self CE only" if args.arm == "c0" else
                            "mean(source self CE, source cross CE) + detached target cross-to-self KL"),
              "cross_context": "all four spatial tokens from every opposite-domain batch sample",
              "shared_bidirectional_attention": True, "alpha": args.alpha,
              "learnable_alpha": args.learnable_alpha,
              "lambda_distill": args.lambda_distill if args.arm == "c1" else 0.0,
              "temperature": 1.0, "inference": "A1 target self branch only",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "source_train_pixels": len(source_train.dataset),
              "source_val_pixels": len(source_val.dataset),
              "target_train_pixels": len(target_train.dataset),
              "checkpoint_selection": f"fixed_epoch_{args.epochs}",
              "target_gt_use": "Strict BiDA GT mask defines target training centers; labels ignored by losses and checkpoint selection",
              "source_split": "BiDA sample_gt random_state=23, stratified 95/5"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = {key: 0.0 for key in ("source_self_ce", "source_cross_ce",
                                           "source_mean_ce", "target_distill_kl", "total",
                                           "target_to_source_entropy", "source_to_target_entropy")}
            count = steps = disagreements = 0
            for (source_images, labels), (target_images, _) in zip(source_train, target_train):
                if len(source_images) != len(target_images):
                    continue
                source_images, labels = source_images.to(device), labels.to(device)
                target_images = target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                if args.arm == "c0":
                    source = model.self_forward(source_images)
                    source_self_ce = F.cross_entropy(source["logits_self"], labels)
                    source_cross_ce = source_self_ce.new_zeros(())
                    source_mean_ce = source_self_ce
                    target_distill = source_self_ce.new_zeros(())
                    loss = source_self_ce
                    target_entropy = source_entropy = source_self_ce.new_zeros(())
                else:
                    output = model(source_images, target_images)
                    source_self_ce = F.cross_entropy(output["source_self_logits"], labels)
                    source_cross_ce = F.cross_entropy(output["source_cross_logits"], labels)
                    source_mean_ce = 0.5 * (source_self_ce + source_cross_ce)
                    target_distill = distillation_kl(output["target_self_logits"],
                                                     output["target_cross_logits"])
                    loss = source_mean_ce + args.lambda_distill * target_distill
                    disagreements += (output["target_self_logits"].argmax(1) !=
                                      output["target_cross_logits"].argmax(1)).sum().item()
                    target_entropy = attention_entropy(output["target_to_source_attention"])
                    source_entropy = attention_entropy(output["source_to_target_attention"])
                loss.backward()
                optimizer.step()
                for key, value in (("source_self_ce", source_self_ce),
                                   ("source_cross_ce", source_cross_ce),
                                   ("source_mean_ce", source_mean_ce),
                                   ("target_distill_kl", target_distill), ("total", loss),
                                   ("target_to_source_entropy", target_entropy),
                                   ("source_to_target_entropy", source_entropy)):
                    totals[key] += value.item() * len(labels)
                count += len(labels)
                steps += 1
            if steps == 0:
                raise RuntimeError("No paired batches with matching size")
            record = {"epoch": epoch, "steps": steps,
                      "losses": {key: value / count for key, value in totals.items()},
                      "alpha": float(model.alpha.detach()),
                      "target_self_cross_disagreement_percent": 100.0 * disagreements / count}
            if epoch == 1 or epoch % 10 == 0:
                record["source_val"] = evaluate_self(model, source_val, device, classes)
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(json.dumps({"epoch": epoch, "losses": record["losses"],
                              "alpha": record["alpha"],
                              "disagreement": record["target_self_cross_disagreement_percent"],
                              "source_val_oa": record.get("source_val", {}).get("oa_percent")}),
                  flush=True)
    checkpoint = args.out / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, checkpoint)
    summary = {"config": config, "checkpoint": str(checkpoint),
               "source_val_at_final": evaluate_self(model, source_val, device, classes),
               "final_alpha": float(model.alpha.detach())}
    if not args.skip_target_eval:
        target, target_classes = target_loader(args.dataset_dir, args.bida_root,
                                               args.seed, args.batch_size,
                                               args.num_workers, args.patch_size)
        if target_classes != classes:
            raise ValueError("Source and target class counts differ")
        summary["target_at_final"] = evaluate_self(model, target, device, classes)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "target_oa": summary.get("target_at_final", {}).get("oa_percent")}), flush=True)


if __name__ == "__main__":
    main()
