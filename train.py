"""Train V0 with paired source/target steps and self-branch inference."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from data import DEFAULT_BIDA_ROOT, DEFAULT_DATA_DIR, source_loaders, target_loader, target_train_loader
from model import DualAxisModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(labels, predictions, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    counts = cm.sum(1)
    recalls = np.divide(np.diag(cm), counts, out=np.zeros(classes, dtype=float), where=counts != 0)
    return {"n": int(cm.sum()), "oa_percent": float(np.trace(cm) / cm.sum() * 100),
            "aa_percent": float(recalls.mean() * 100),
            "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
            "per_class_percent": (recalls * 100).tolist(), "confusion_matrix": cm.tolist()}


def memory_to_self_kl(logits_self, logits_memory, temperature=1.0):
    """Teacher is detached; gradients enter only the target self prediction."""
    teacher = F.softmax(logits_memory.detach() / temperature, dim=1)
    student = F.log_softmax(logits_self / temperature, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature ** 2


def source_ce_loss(logits_self, labels, logits_memory=None):
    self_ce = F.cross_entropy(logits_self, labels)
    if logits_memory is None:
        return self_ce, self_ce, self_ce.new_zeros(())
    memory_ce = F.cross_entropy(logits_memory, labels)
    return 0.5 * (self_ce + memory_ce), self_ce, memory_ce


@torch.inference_mode()
def evaluate(model, loader, device, classes, diagnostics=False):
    model.eval()
    labels, predictions, base_predictions = [], [], []
    attention_sums = {key: None for key in ("spatial_attn", "spectral_attn",
                       "class_attn_self", "memory_spatial_attn", "memory_spectral_attn")}
    attention_counts = {key: 0 for key in attention_sums}
    for images, target in loader:
        images = images.to(device)
        result = model(images)
        predictions.append(result["logits_self"].argmax(1).cpu().numpy())
        labels.append(target.numpy())
        if diagnostics and model.variant == "a3":
            base_predictions.append(result["logits_memory"].argmax(1).cpu().numpy())
        if diagnostics:
            for key in attention_sums:
                value = result[key]
                if value is not None:
                    value = value.sum(0).cpu()
                    attention_sums[key] = value if attention_sums[key] is None else attention_sums[key] + value
                    attention_counts[key] += images.shape[0]
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    report = metrics(labels, predictions, classes)
    if diagnostics:
        for key in attention_sums:
            mean = attention_sums[key]
            report["mean_" + key] = (mean / attention_counts[key]).tolist() if mean is not None else None
        if model.variant == "a2_lite":
            class_attention = report["mean_class_attn_self"]
            report["class_attention_spatial_mass"] = float(sum(class_attention[:4]))
            report["class_attention_spectral_mass"] = float(sum(class_attention[4:]))
        if base_predictions:
            memory_predictions = np.concatenate(base_predictions)
            changed = memory_predictions != predictions
            report["memory_change"] = {
                "changed_percent": float(changed.mean() * 100),
                "self_correct_to_memory_wrong": int(((predictions == labels) & (memory_predictions != labels)).sum()),
                "self_wrong_to_memory_correct": int(((predictions != labels) & (memory_predictions == labels)).sum()),
            }
            report["alpha_spatial"] = float(model.alpha_spatial.item())
            report["alpha_spectral"] = float(model.alpha_spectral.item())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("a0", "a1", "a2_lite", "a2", "a3"), required=True)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bida-root", type=Path, default=DEFAULT_BIDA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-target-eval", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    train, val, classes, bands = source_loaders(args.dataset_dir, args.bida_root, args.seed,
                                                  args.batch_size, args.num_workers, args.patch_size)
    target_train = target_train_loader(args.dataset_dir, args.bida_root, args.seed,
                                       args.batch_size, args.num_workers, args.patch_size,
                                       generator=train.generator)
    model = DualAxisModel(bands=bands, classes=classes, patch_size=args.patch_size, variant=args.variant).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    config = {"variant": args.variant, "seed": args.seed, "epochs": args.epochs,
              "batch_size": args.batch_size, "patch_size": args.patch_size, "lr": args.lr,
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              "optimizer": "SGD, no momentum or schedule (Strict BiDA)",
              "source_train_pixels": len(train.dataset), "source_val_pixels": len(val.dataset),
              "target_train_pixels": len(target_train.dataset),
              "checkpoint_selection": f"fixed_epoch_{args.epochs}",
              "target_gt_use": "Strict BiDA target GT mask selects unlabeled training centers; label values ignored by loss; metrics after training",
              "source_ce_weights": [0.5, 0.5] if args.variant == "a3" else [1.0],
              "inference_branch": "target_self",
              "source_split": "BiDA sample_gt random_state=23, stratified 95/5",
              "dataset_dir": str(args.dataset_dir), "bida_root": str(args.bida_root)}
    if args.variant == "a3":
        config.update({"a3_training": "mean(source self CE, source memory CE) + detached target memory teacher KL; EMA updated after each optimizer step",
                       "lambda_dist": 1.0, "temperature": 1.0})
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sums = {"source_self_ce": 0.0, "source_memory_ce": 0.0,
                         "source_mean_ce": 0.0, "target_distill_kl": 0.0,
                         "total": 0.0}
            count = 0
            steps = 0
            target_disagreements = 0
            for (images, labels), (target_images, _) in zip(train, target_train):
                if images.shape[0] != target_images.shape[0]:
                    continue
                images, labels = images.to(device), labels.to(device)
                target_images = target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                source = model(images)
                if args.variant == "a3":
                    target = model(target_images)
                    loss_source_mean, loss_self, loss_memory = source_ce_loss(
                        source["logits_self"], labels, source["logits_memory"])
                    loss_distill = memory_to_self_kl(target["logits_self"], target["logits_memory"])
                    loss = loss_source_mean + loss_distill
                    target_disagreements += (target["logits_self"].argmax(1) !=
                                             target["logits_memory"].argmax(1)).sum().item()
                else:
                    loss_source_mean, loss_self, loss_memory = source_ce_loss(source["logits_self"], labels)
                    loss_distill = loss_self.new_zeros(())
                    loss = loss_source_mean
                loss.backward()
                optimizer.step()
                if args.variant == "a3":
                    model.update_memory(source["spatial_tokens"], source["spectral_tokens"], labels)
                for key, value in (("source_self_ce", loss_self), ("source_memory_ce", loss_memory),
                                   ("source_mean_ce", loss_source_mean),
                                   ("target_distill_kl", loss_distill), ("total", loss)):
                    loss_sums[key] += value.item() * len(labels)
                count += len(labels)
                steps += 1
            if steps == 0:
                raise RuntimeError("No paired source/target batches with matching sizes")
            record = {"epoch": epoch, "steps": steps,
                      "losses": {key: value / count for key, value in loss_sums.items()}}
            if args.variant == "a3":
                record["alpha_spatial"] = float(model.alpha_spatial.detach().item())
                record["alpha_spectral"] = float(model.alpha_spectral.detach().item())
                record["target_self_memory_disagreement_percent"] = 100.0 * target_disagreements / count
            if epoch % 10 == 0 or epoch == 1:
                record["source_val"] = evaluate(model, val, device, classes)
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(json.dumps({"epoch": epoch, "losses": record["losses"],
                              "alpha_spatial": record.get("alpha_spatial"),
                              "alpha_spectral": record.get("alpha_spectral"),
                              "target_self_memory_disagreement_percent": record.get("target_self_memory_disagreement_percent"),
                              "source_val_oa": record.get("source_val", {}).get("oa_percent")}), flush=True)
    checkpoint = args.out / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, checkpoint)
    summary = {"config": config, "checkpoint": str(checkpoint)}
    if not args.skip_target_eval:
        target, target_classes = target_loader(args.dataset_dir, args.bida_root, args.seed,
                                                args.batch_size, args.num_workers, args.patch_size)
        if target_classes != classes:
            raise ValueError("Source and target class counts differ")
        summary["target_at_final"] = evaluate(model, target, device, classes, diagnostics=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
