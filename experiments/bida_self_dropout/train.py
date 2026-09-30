"""BiDA-self dropout-0 intervention with frozen 10-epoch target trajectory."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BIDA_ROOT = Path("/home/zhangzj26/IEEE_TCSVT_BiDA")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIDA_ROOT))
sys.path.insert(0, str(BIDA_ROOT / "experiments/bida_memory_v1"))
from experiments.bida_memory_v1.run import make_loaders  # noqa: E402
from models.get_model import get_model  # noqa: E402
from utils.scheduler import load_scheduler  # noqa: E402
from utils.utils_HSI import seed_worker  # noqa: E402

EXPECTED_HASH = "dadb50a683a2bab506df66c317cbea8d08aee955bab5281013dd4604d6fa40b5"
CONTROL_RESULT = PROJECT_ROOT / "results/bida_self/diagnostic_2100/results.json"


def state_hash(model):
    h = hashlib.sha256()
    for value in model.state_dict().values():
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def source_evaluate(model, loader, device):
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, images)[1]
        loss_sum += F.cross_entropy(logits, labels).item() * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    return {"oa_percent": correct / count * 100, "loss": loss_sum / count}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    outputs = []
    for images, _ in loader:
        images = images.to(device)
        outputs.append(model(images, images)[1].cpu().numpy())
    logits = np.concatenate(outputs).astype(np.float32, copy=False)
    return logits, logits.argmax(1).astype(np.int64, copy=False)


def metrics(labels, predictions, logits, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(classes, dtype=float), where=cm.sum(1) != 0)
    counts = np.bincount(predictions, minlength=classes)
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    return {"n": int(len(labels)), "oa_percent": float(np.trace(cm) / cm.sum() * 100),
            "aa_percent": float(recall.mean() * 100),
            "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
            "per_class_percent": (recall * 100).tolist(), "prediction_counts": counts.tolist(),
            "prediction_shares": (counts / counts.sum()).tolist(),
            "mean_entropy": float(-(probability * np.log(np.clip(probability, 1e-12, 1))).sum(1).mean()),
            "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="4")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-dir", default="/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out / "checkpoints"
    checkpoint_dir.mkdir()
    prediction_dir = args.out / "predictions"
    prediction_dir.mkdir()
    args.model, args.lr, args.num_tokens, args.dim, args.depth = "BiDA", 0.01, 4, 64, 3
    args.loss_type, args.labelsmooth = "softmax", "off"
    device = torch.device("cuda:" + args.device)
    control = json.loads(CONTROL_RESULT.read_text())["target_at_final"]

    seed_worker(args.seed)
    train_loader, val_loader, target_train_loader, target_eval_loader, _, classes = make_loaders(args)
    model = get_model("BiDA", "Houston13", 13, args)
    initial_hash = state_hash(model)
    if initial_hash != EXPECTED_HASH:
        raise RuntimeError(f"Initial state mismatch: {initial_hash}")
    unused_ema = get_model("BiDA", "Houston13", 13, args, ema=True)
    del unused_ema
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            dropout_count += 1
    model = model.to(device)
    optimizer, scheduler = load_scheduler("BiDA", model, args)
    config = {"arm": "bida_self_dropout0", "seed": args.seed, "epochs": args.epochs,
              "initial_model_sha256_before_dropout_change": initial_hash,
              "dropout_modules_changed": dropout_count, "dropout": 0.0, "depth": 3,
              "bn": "mixed source/target", "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "objective": "source self CE only", "primary": "fixed_epoch_200",
              "secondary": "target-oracle max over frozen epochs 10:10:200",
              "protocol_sha256": file_hash(Path(__file__).with_name("PROTOCOL.md"))}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.monotonic()
    checkpoint_records = []
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, count, steps = 0.0, 0, 0
            for (source_images, labels), (target_images, _) in zip(train_loader, target_train_loader):
                if len(source_images) != len(target_images):
                    continue
                source_images, labels = source_images.to(device), labels.to(device)
                target_images = target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                source_logits, _, _, _ = model(source_images, target_images)
                loss = F.cross_entropy(source_logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
                steps += 1
            if scheduler is not None:
                scheduler.step()
            record = {"epoch": epoch, "steps": steps, "source_ce": loss_sum / count}
            if epoch == 1 or epoch % 10 == 0:
                record["source_val"] = source_evaluate(model, val_loader, device)
            if epoch % 10 == 0:
                path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, path)
                checkpoint_records.append({"epoch": epoch, "path": str(path), "sha256": file_hash(path)})
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(record), flush=True)
    train_seconds = time.monotonic() - start

    # Freeze predictions for every official-grid checkpoint before labels are collected.
    prediction_records = []
    inference_start = time.monotonic()
    for item in checkpoint_records:
        saved = torch.load(item["path"], map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        logits, predictions = predict(model, target_eval_loader, device)
        path = prediction_dir / f"epoch_{item['epoch']:03d}.npz"
        np.savez_compressed(path, logits=logits, predictions=predictions)
        prediction_records.append({"epoch": item["epoch"], "path": str(path), "sha256": file_hash(path)})
        print(json.dumps({"frozen_prediction_epoch": item["epoch"], "sha256": prediction_records[-1]["sha256"]}), flush=True)
    inference_seconds = time.monotonic() - inference_start
    manifest = {"config": config, "checkpoints": checkpoint_records, "predictions": prediction_records,
                "train_seconds": train_seconds, "trajectory_inference_seconds": inference_seconds}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_sha256 = file_hash(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_sha256}), flush=True)

    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in target_eval_loader])
    trajectory = []
    for item in prediction_records:
        artifact = np.load(item["path"])
        trajectory.append({"epoch": item["epoch"],
                           "metrics": metrics(labels, artifact["predictions"], artifact["logits"], classes)})
    final = trajectory[-1]["metrics"]
    best = max(trajectory, key=lambda row: row["metrics"]["oa_percent"])
    result = {"frozen_manifest_sha256": manifest_sha256, "target_at_final": final,
              "target_oracle": {"best_epoch": best["epoch"], "metrics": best["metrics"]},
              "target_trajectory": trajectory, "dropout01_control_final": control,
              "dropout01_minus_dropout0_final": {"oa_points": control["oa_percent"] - final["oa_percent"],
                                                  "aa_points": control["aa_percent"] - final["aa_percent"],
                                                  "kappa": control["kappa"] - final["kappa"]},
              "train_seconds": train_seconds, "trajectory_inference_seconds": inference_seconds}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({"target_at_final": final, "target_oracle": result["target_oracle"],
                                  "dropout01_minus_dropout0_final": result["dropout01_minus_dropout0_final"]}), flush=True)


if __name__ == "__main__":
    main()
