"""Paired A1 control and source-only spatial attention diversity experiment."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.slot_uda.train import evaluate, set_seed
from experiments.v0.data import (DEFAULT_BIDA_ROOT, DEFAULT_DATA_DIR, source_loaders,
                                 target_loader, target_train_loader)
from experiments.v0.model import DualAxisModel


ARMS = ("d0", "d1")


def diversity_loss(attention):
    """Mean squared off-diagonal cosine of the four spatial attention maps.

    This is ||A_hat A_hat^T - I||_F^2 / [K(K-1)], averaged over samples.
    A_hat is L2-normalized over the 169 spatial positions. The denominator
    keeps the coefficient independent of the number of query slots.
    """
    if attention.ndim != 3 or attention.shape[1] < 2:
        raise ValueError("Expected attention [batch, slots>=2, spatial positions]")
    normalized = F.normalize(attention, p=2, dim=-1)
    gram = normalized @ normalized.transpose(1, 2)
    slots = attention.shape[1]
    identity = torch.eye(slots, device=attention.device, dtype=attention.dtype)
    return (gram - identity).square().sum(dim=(1, 2)).mean() / (slots * (slots - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-div", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bida-root", type=Path, default=DEFAULT_BIDA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-target-eval", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lambda_div < 0:
        parser.error("epochs must be positive and lambda-div nonnegative")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    source_train, source_val, classes, bands = source_loaders(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size,
        args.num_workers, args.patch_size)
    # Preserve A1's paired-loader iteration and source batch order. Target
    # images and labels never enter either D0 or D1 loss.
    target_train = target_train_loader(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size,
        args.num_workers, args.patch_size, generator=source_train.generator)
    model = DualAxisModel(bands=bands, classes=classes, patch_size=args.patch_size,
                          variant="a1").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    weight = args.lambda_div if args.arm == "d1" else 0.0
    config = {"arm": args.arm, "backbone": "v0_a1", "seed": args.seed,
              "epochs": args.epochs, "batch_size": args.batch_size,
              "patch_size": args.patch_size, "lr": args.lr,
              "optimizer": "SGD, no momentum or schedule (Strict BiDA)",
              "diversity": "source attention L2-normalized over spatial positions; mean squared off-diagonal cosine",
              "lambda_div": weight, "diversity_on": "labeled source training patches only",
              "source_train_pixels": len(source_train.dataset),
              "source_val_pixels": len(source_val.dataset),
              "target_train_pixels": len(target_train.dataset),
              "checkpoint_selection": f"fixed_epoch_{args.epochs}",
              "target_gt_use": "Strict BiDA GT mask defines target training candidate centers; target labels do not enter D0/D1 losses or checkpoint selection",
              "source_split": "BiDA sample_gt random_state=23, stratified 95/5",
              "dataset_dir": str(args.dataset_dir), "bida_root": str(args.bida_root)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals = {"source_ce": 0.0, "diversity": 0.0,
                      "weighted_diversity": 0.0, "total": 0.0}
            count = steps = 0
            for (images, labels), (target_images, _) in zip(source_train, target_train):
                if len(images) != len(target_images):
                    continue
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                source = model(images)
                loss_ce = F.cross_entropy(source["logits_self"], labels)
                loss_div = diversity_loss(source["spatial_attn"])
                loss = loss_ce + weight * loss_div
                loss.backward()
                optimizer.step()
                for key, value in (("source_ce", loss_ce), ("diversity", loss_div),
                                   ("weighted_diversity", weight * loss_div),
                                   ("total", loss)):
                    totals[key] += value.item() * len(labels)
                count += len(labels)
                steps += 1
            if steps == 0:
                raise RuntimeError("No paired source/target batches with matching sizes")
            record = {"epoch": epoch, "steps": steps,
                      "losses": {key: value / count for key, value in totals.items()}}
            if epoch == 1 or epoch % 10 == 0:
                record["source_val"] = evaluate(model, source_val, device, classes,
                                                diagnostics=True)
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(json.dumps({"epoch": epoch, "losses": record["losses"],
                              "source_val_oa": record.get("source_val", {}).get("oa_percent")}),
                  flush=True)
    checkpoint = args.out / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, checkpoint)
    summary = {"config": config, "checkpoint": str(checkpoint),
               "source_val_at_final": evaluate(model, source_val, device,
                                               classes, diagnostics=True)}
    if not args.skip_target_eval:
        target, target_classes = target_loader(args.dataset_dir, args.bida_root,
                                               args.seed, args.batch_size,
                                               args.num_workers, args.patch_size)
        if target_classes != classes:
            raise ValueError("Source and target class counts differ")
        summary["target_at_final"] = evaluate(model, target, device,
                                              classes, diagnostics=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"arm": args.arm, "seed": args.seed,
                      "target_oa": summary.get("target_at_final", {}).get("oa_percent")}),
          flush=True)


if __name__ == "__main__":
    main()
