"""A1 spatial-slot UDA controls under the frozen Houston BiDA protocol."""

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
from experiments.v0.model import DualAxisModel


ARMS = {"b1": ("global", False), "b2": ("slot", False),
        "b3": (None, True), "b4": ("slot", True)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mmd2(source, target):
    """Biased, nonnegative multi-kernel RBF MMD squared for [batch, dim] tokens.

    The detached median squared distance sets the common kernel convention.
    Both domains receive gradients; no arbitrary sample pairing is used.
    """
    joined = torch.cat((source, target), dim=0)
    distance2 = torch.cdist(joined, joined).square()
    with torch.no_grad():
        upper = torch.triu_indices(len(joined), len(joined), offset=1, device=joined.device)
        bandwidth = distance2[upper[0], upper[1]].median().clamp_min(1e-6)
    kernel = sum(torch.exp(-distance2 / (bandwidth * scale)) for scale in (0.5, 1.0, 2.0)) / 3.0
    n = len(source)
    return (kernel[:n, :n].mean() + kernel[n:, n:].mean() -
            2.0 * kernel[:n, n:].mean()).clamp_min(0.0)


def alignment_loss(source_slots, target_slots, alignment):
    if alignment == "global":
        return mmd2(source_slots.mean(dim=1), target_slots.mean(dim=1))
    if alignment == "slot":
        return torch.stack([mmd2(source_slots[:, k], target_slots[:, k])
                            for k in range(source_slots.shape[1])]).mean()
    return source_slots.new_zeros(())


def strong_view(weak):
    """A second center-preserving HSI view; no spectral mixing or target labels."""
    rotations = int(torch.randint(1, 4, (), device=weak.device).item())
    strong = torch.rot90(weak, rotations, dims=(-2, -1))
    if bool(torch.randint(0, 2, (), device=weak.device).item()):
        strong = torch.flip(strong, dims=(-1,))
    return strong


def consistency_loss(weak_logits, strong_logits):
    teacher = F.softmax(weak_logits.detach(), dim=1)
    return F.kl_div(F.log_softmax(strong_logits, dim=1), teacher, reduction="batchmean")


def metrics(labels, predictions, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    counts = cm.sum(1)
    recalls = np.divide(np.diag(cm), counts, out=np.zeros(classes, dtype=float), where=counts != 0)
    return {"n": int(cm.sum()), "oa_percent": float(np.trace(cm) / cm.sum() * 100),
            "aa_percent": float(recalls.mean() * 100),
            "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
            "per_class_percent": (recalls * 100).tolist(),
            "prediction_counts": np.bincount(predictions, minlength=classes).tolist(),
            "confusion_matrix": cm.tolist()}


@torch.inference_mode()
def evaluate(model, loader, device, classes, diagnostics=False):
    model.eval()
    labels, predictions = [], []
    attention_sum = None
    cosine_sum = None
    attention_cosine_sum = None
    count = 0
    for images, truth in loader:
        output = model(images.to(device))
        labels.append(truth.numpy())
        predictions.append(output["logits_self"].argmax(1).cpu().numpy())
        if diagnostics:
            attention = output["spatial_attn"].sum(0).cpu()
            slots = F.normalize(output["spatial_tokens"], dim=-1)
            cosine = (slots @ slots.transpose(1, 2)).sum(0).cpu()
            attention_unit = F.normalize(output["spatial_attn"], dim=-1)
            attention_cosine = (attention_unit @ attention_unit.transpose(1, 2)).sum(0).cpu()
            attention_sum = attention if attention_sum is None else attention_sum + attention
            cosine_sum = cosine if cosine_sum is None else cosine_sum + cosine
            attention_cosine_sum = (attention_cosine if attention_cosine_sum is None
                                    else attention_cosine_sum + attention_cosine)
            count += len(images)
    report = metrics(np.concatenate(labels), np.concatenate(predictions), classes)
    if diagnostics:
        report["mean_spatial_attn"] = (attention_sum / count).tolist()
        report["mean_slot_cosine"] = (cosine_sum / count).tolist()
        report["mean_attention_cosine"] = (attention_cosine_sum / count).tolist()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-cons", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bida-root", type=Path, default=DEFAULT_BIDA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-target-eval", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lambda_align < 0 or args.lambda_cons < 0:
        parser.error("epochs must be positive and loss weights nonnegative")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    alignment, use_consistency = ARMS[args.arm]
    source_train, source_val, classes, bands = source_loaders(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size,
        args.num_workers, args.patch_size)
    target_train = target_train_loader(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size,
        args.num_workers, args.patch_size, generator=source_train.generator)
    model = DualAxisModel(bands=bands, classes=classes, patch_size=args.patch_size,
                          variant="a1").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    config = {"arm": args.arm, "backbone": "v0_a1", "seed": args.seed,
              "epochs": args.epochs, "batch_size": args.batch_size,
              "patch_size": args.patch_size, "lr": args.lr,
              "optimizer": "SGD, no momentum or schedule (Strict BiDA)",
              "alignment": alignment, "mmd": "biased RBF MMD squared; detached batch median bandwidth; scales 0.5,1,2; slot mean",
              "consistency": "KL(stopgrad(weak prediction) || strong prediction), T=1; strong view is rotation plus optional flip",
              "lambda_align": args.lambda_align, "lambda_cons": args.lambda_cons,
              "source_train_pixels": len(source_train.dataset),
              "source_val_pixels": len(source_val.dataset),
              "target_train_pixels": len(target_train.dataset),
              "checkpoint_selection": f"fixed_epoch_{args.epochs}",
              "target_gt_use": "Strict BiDA GT mask selects target training centers; labels ignored by losses and selection",
              "source_split": "BiDA sample_gt random_state=23, stratified 95/5",
              "dataset_dir": str(args.dataset_dir), "bida_root": str(args.bida_root)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = {"source_ce": 0.0, "alignment_mmd2": 0.0,
                      "target_consistency_kl": 0.0, "total": 0.0}
            count = steps = 0
            for (source_images, labels), (target_images, _) in zip(source_train, target_train):
                if len(source_images) != len(target_images):
                    continue
                source_images = source_images.to(device)
                labels = labels.to(device)
                target_images = target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                source = model(source_images)
                loss_ce = F.cross_entropy(source["logits_self"], labels)
                if alignment or use_consistency:
                    target_weak = model(target_images)
                loss_align = (alignment_loss(source["spatial_tokens"],
                                             target_weak["spatial_tokens"], alignment)
                              if alignment else loss_ce.new_zeros(()))
                if use_consistency:
                    target_strong = model(strong_view(target_images))
                    loss_cons = consistency_loss(target_weak["logits_self"],
                                                 target_strong["logits_self"])
                else:
                    loss_cons = loss_ce.new_zeros(())
                loss = loss_ce + args.lambda_align * loss_align + args.lambda_cons * loss_cons
                loss.backward()
                optimizer.step()
                for key, value in (("source_ce", loss_ce), ("alignment_mmd2", loss_align),
                                   ("target_consistency_kl", loss_cons), ("total", loss)):
                    totals[key] += value.item() * len(labels)
                count += len(labels)
                steps += 1
            if steps == 0:
                raise RuntimeError("No paired source/target batches with matching sizes")
            record = {"epoch": epoch, "steps": steps,
                      "losses": {key: value / count for key, value in totals.items()}}
            if epoch == 1 or epoch % 10 == 0:
                record["source_val"] = evaluate(model, source_val, device, classes)
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(json.dumps({"epoch": epoch, "losses": record["losses"],
                              "source_val_oa": record.get("source_val", {}).get("oa_percent")}), flush=True)
    checkpoint = args.out / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, checkpoint)
    summary = {"config": config, "checkpoint": str(checkpoint),
               "source_val_at_final": evaluate(model, source_val, device, classes, diagnostics=True)}
    if not args.skip_target_eval:
        target, target_classes = target_loader(args.dataset_dir, args.bida_root,
                                               args.seed, args.batch_size,
                                               args.num_workers, args.patch_size)
        if target_classes != classes:
            raise ValueError("Source and target class counts differ")
        summary["target_at_final"] = evaluate(model, target, device, classes, diagnostics=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "target_oa": summary.get("target_at_final", {}).get("oa_percent")}), flush=True)


if __name__ == "__main__":
    main()
