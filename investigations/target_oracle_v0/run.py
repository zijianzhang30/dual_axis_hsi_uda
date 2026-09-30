"""Independent V0 rerun with released-BiDA target-OA oracle selection."""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
V0 = PROJECT / "experiments" / "v0"
sys.path.insert(0, str(V0))

from data import DEFAULT_BIDA_ROOT, DEFAULT_DATA_DIR, source_loaders, target_loader, target_train_loader
from model import DualAxisModel
from train import evaluate, set_seed, source_ce_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("a0", "a1", "a2"), required=True)
    parser.add_argument("--seed", type=int, choices=(2100, 2101, 2102), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bida-root", type=Path, default=DEFAULT_BIDA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    original_dir = PROJECT / "results" / "v0" / f"{args.variant}_{args.seed}"
    original = [json.loads(line) for line in (original_dir / "history.jsonl").read_text().splitlines()]
    if len(original) != 200 or original[-1]["epoch"] != 200:
        raise ValueError("Expected completed 200-epoch V0 history")
    set_seed(args.seed)
    device = torch.device(args.device)
    train, val, classes, bands = source_loaders(args.dataset_dir, args.bida_root, args.seed,
                                                  128, args.num_workers, 13)
    target_train = target_train_loader(args.dataset_dir, args.bida_root, args.seed,
                                       128, args.num_workers, 13, generator=train.generator)
    model = DualAxisModel(bands=bands, classes=classes, patch_size=13, variant=args.variant).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir()
    max_train_loss_delta = 0.0
    source_val_mismatches = 0
    with (args.out / "replay_audit.jsonl").open("w") as audit:
        for epoch in range(1, 201):
            model.train()
            loss_sum = 0.0
            count = 0
            steps = 0
            for (images, labels), (target_images, _) in zip(train, target_train):
                if images.shape[0] != target_images.shape[0]:
                    continue
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                source = model(images)
                loss, _, _ = source_ce_loss(source["logits_self"], labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
                steps += 1
            record = {"epoch": epoch, "steps": steps, "source_ce": loss_sum / count}
            expected = original[epoch - 1]
            if steps != expected["steps"]:
                raise RuntimeError(f"Paired batch count differs at epoch {epoch}")
            record["original_source_ce"] = expected["losses"]["total"]
            delta = abs(record["source_ce"] - record["original_source_ce"])
            max_train_loss_delta = max(max_train_loss_delta, delta)
            if epoch == 1 or epoch % 10 == 0:
                record["source_val_oa_percent"] = evaluate(model, val, device, classes)["oa_percent"]
                if record["source_val_oa_percent"] != expected["source_val"]["oa_percent"]:
                    source_val_mismatches += 1
            audit.write(json.dumps(record) + "\n")
            audit.flush()
            if epoch % 10 == 0:
                torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoints / f"epoch_{epoch:03d}.pth")
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(record), flush=True)

    target, target_classes = target_loader(args.dataset_dir, args.bida_root, args.seed,
                                            128, args.num_workers, 13)
    if target_classes != classes:
        raise ValueError("Source and target class counts differ")
    epoch_metrics = []
    best_target = -1.0
    best_epoch = None
    with (args.out / "epoch_metrics.jsonl").open("w") as metrics_file:
        for epoch in range(10, 201, 10):
            path = checkpoints / f"epoch_{epoch:03d}.pth"
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
            result = evaluate(model, target, device, classes)
            record = {"epoch": epoch, "target": result}
            epoch_metrics.append(record)
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            if result["oa_percent"] > best_target:
                best_target = result["oa_percent"]
                best_epoch = epoch
            print(json.dumps({"epoch": epoch, "target_oa_percent": result["oa_percent"],
                              "best_epoch": best_epoch, "best_target_oa_percent": best_target}), flush=True)
    frozen = json.loads((original_dir / "summary.json").read_text())
    winner = next(record for record in epoch_metrics if record["epoch"] == best_epoch)
    report = {"variant": args.variant, "seed": args.seed,
              "selection": "maximum_Houston18_OA_among_epochs_10_to_200_step_10; target-oracle diagnostic",
              "selected_epoch": best_epoch, "target_at_best": winner["target"],
              "target_at_rerun_final": epoch_metrics[-1]["target"],
              "target_at_original_final": frozen["target_at_final"],
              "max_train_loss_difference_from_original": max_train_loss_delta,
              "source_val_checkpoints_different_from_original": source_val_mismatches,
              "checkpoint": str(checkpoints / f"epoch_{best_epoch:03d}.pth"),
              "trajectory": "independent rerun; original trajectory was not exactly reproducible"}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps({"variant": args.variant, "seed": args.seed,
                                    "epoch": best_epoch, "target_oa": best_target,
                                    "rerun_final_oa": epoch_metrics[-1]["target"]["oa_percent"]}), flush=True)


if __name__ == "__main__":
    main()
