"""H2: released BiDA representation with source CE as its only loss."""

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


EXPECTED_INITIAL_HASH = "dadb50a683a2bab506df66c317cbea8d08aee955bab5281013dd4604d6fa40b5"


def state_hash(model):
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def source_evaluate(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
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
    logits = []
    for images, _ in loader:
        images = images.to(device)
        logits.append(model(images, images)[1].cpu().numpy())
    logits = np.concatenate(logits).astype(np.float32, copy=False)
    return logits, logits.argmax(1).astype(np.int64, copy=False)


def metrics(labels, predictions, logits, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(classes, dtype=float), where=cm.sum(1) != 0)
    counts = np.bincount(predictions, minlength=classes)
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    return {
        "n": int(len(labels)),
        "oa_percent": float(np.trace(cm) / cm.sum() * 100),
        "aa_percent": float(recall.mean() * 100),
        "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
        "per_class_percent": (recall * 100).tolist(),
        "prediction_counts": counts.tolist(),
        "prediction_shares": (counts / counts.sum()).tolist(),
        "mean_entropy": float(-(probability * np.log(np.clip(probability, 1e-12, 1))).sum(1).mean()),
        "confusion_matrix": cm.tolist(),
    }


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
    args.model = "BiDA"
    args.lr = 0.01
    args.num_tokens = 4
    args.dim = 64
    args.depth = 3
    args.loss_type = "softmax"
    args.labelsmooth = "off"
    device = torch.device("cuda:" + args.device)

    seed_worker(args.seed)
    train_loader, val_loader, target_train_loader, target_eval_loader, _, classes = make_loaders(args)
    model = get_model("BiDA", "Houston13", 13, args)
    initial_hash = state_hash(model)
    if args.seed == 2100 and initial_hash != EXPECTED_INITIAL_HASH:
        raise RuntimeError(f"Initial state mismatch: {initial_hash}")
    # Preserve released initialization RNG consumption. This model is never used.
    unused_ema = get_model("BiDA", "Houston13", 13, args, ema=True)
    del unused_ema
    model = model.to(device)
    optimizer, scheduler = load_scheduler("BiDA", model, args)
    config = {
        "arm": "bida_self",
        "seed": args.seed,
        "epochs": args.epochs,
        "initial_model_sha256": initial_hash,
        "source_train_pixels": len(train_loader.dataset),
        "source_val_pixels": len(val_loader.dataset),
        "target_train_pixels": len(target_train_loader.dataset),
        "target_eval_pixels": len(target_eval_loader.dataset),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": "SGD lr=0.01, no momentum or schedule",
        "objective": "source self CE only",
        "target_training_role": "forwarded after source inside unmodified BiDA; updates shared BN; no target loss",
        "disabled": ["coupled distillation", "EMA consistency", "MMD"],
        "checkpoint_selection": f"fixed_epoch_{args.epochs}",
        "target_gt_use": "GT mask selects unlabeled centers; values excluded from loss and selection",
        "protocol_sha256": file_hash(Path(__file__).with_name("PROTOCOL.md")),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.monotonic()
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            count = 0
            steps = 0
            for (source_images, labels), (target_images, _) in zip(train_loader, target_train_loader):
                if len(source_images) != len(target_images):
                    continue
                source_images = source_images.to(device)
                labels = labels.to(device)
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
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(record), flush=True)

    train_seconds = time.monotonic() - start
    checkpoint = args.out / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, checkpoint)
    inference_start = time.monotonic()
    logits, predictions = predict(model, target_eval_loader, device)
    inference_seconds = time.monotonic() - inference_start
    prediction_file = args.out / "predictions.npz"
    np.savez_compressed(prediction_file, logits=logits, predictions=predictions)
    frozen = {
        "config": config,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "prediction_file": str(prediction_file),
        "prediction_sha256": file_hash(prediction_file),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
    }
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(frozen, indent=2) + "\n")
    manifest_sha256 = file_hash(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_sha256, **frozen}), flush=True)

    # Post-hoc target metrics only after predictions and hashes are frozen.
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in target_eval_loader])
    report = metrics(labels, predictions, logits, classes)
    strict_oa = 78.75276459802272
    strict_aa = 66.94826760876387
    a1_oa = 73.58083968166953
    a1_aa = 62.12991349505748
    result = {
        "frozen_manifest_sha256": manifest_sha256,
        "target_at_final": report,
        "comparisons": {
            "bida_self_minus_a1_oa_points": report["oa_percent"] - a1_oa,
            "bida_self_minus_a1_aa_points": report["aa_percent"] - a1_aa,
            "strict_minus_bida_self_oa_points": strict_oa - report["oa_percent"],
            "strict_minus_bida_self_aa_points": strict_aa - report["aa_percent"],
            "within_2pp_of_strict": strict_oa - report["oa_percent"] < 2.0,
            "strict_advantage_at_least_2pp": strict_oa - report["oa_percent"] >= 2.0,
        },
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
