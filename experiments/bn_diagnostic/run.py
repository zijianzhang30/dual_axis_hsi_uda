"""Frozen H1 diagnostic: change only Strict BiDA BatchNorm running statistics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BIDA_ROOT = Path("/home/zhangzj26/IEEE_TCSVT_BiDA")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BIDA_ROOT))

from models.BiDA import BiDA  # noqa: E402
from utils.dataset import HSIDataset, load_mat_hsi, sample_gt  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_dataset(name, dataset_dir, source_train=False):
    image, gt, labels = load_mat_hsi(name, str(dataset_dir) + "/", norm="normband")
    if source_train:
        gt, _ = sample_gt(gt, 0.95, 2100, mode="random")
    else:
        gt, _ = sample_gt(gt, 1, 2100, mode="random")
    radius = 6
    image = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    gt = np.pad(gt, ((radius, radius), (radius, radius)), mode="reflect")
    return HSIDataset(image, gt, 13, data_aug=False), len(labels)


def make_loader(dataset, batch_size, workers):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        drop_last=False, generator=torch.Generator().manual_seed(2100)
    )


def build_model(checkpoint, device):
    class Options:
        num_tokens = 4
        dim = 64
        depth = 3

    model = BiDA("Houston13", Options())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("epoch") != 200:
        raise ValueError("H1 requires the fixed epoch-200 checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device)


@torch.inference_mode()
def recalibrate(model, loader, device):
    model.eval()
    batch_norms = [module for module in model.modules()
                   if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    count = 0
    for images, _ in loader:
        images = images.to(device)
        model._tokenize(images)
        count += len(images)
    model.eval()
    return count


def bn_snapshot(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            result[name] = {
                "running_mean": module.running_mean.detach().cpu().tolist(),
                "running_var": module.running_var.detach().cpu().tolist(),
                "num_batches_tracked": int(module.num_batches_tracked.item()),
            }
    return result


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits = []
    for images, _ in loader:
        images = images.to(device)
        logits.append(model(images, images)[1].cpu().numpy())
    logits = np.concatenate(logits).astype(np.float32, copy=False)
    return logits, logits.argmax(1).astype(np.int64, copy=False)


def target_labels(loader):
    return np.concatenate([labels.numpy() for _, labels in loader]).astype(np.int64, copy=False)


def metrics(labels, predictions, logits, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    counts = cm.sum(1)
    recall = np.divide(np.diag(cm), counts, out=np.zeros(classes, dtype=float), where=counts != 0)
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(1)
    predicted = np.bincount(predictions, minlength=classes)
    return {
        "n": int(len(labels)),
        "oa_percent": float(np.trace(cm) / cm.sum() * 100),
        "aa_percent": float(recall.mean() * 100),
        "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
        "per_class_percent": (recall * 100).tolist(),
        "prediction_counts": predicted.tolist(),
        "prediction_shares": (predicted / predicted.sum()).tolist(),
        "mean_entropy": float(entropy.mean()),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston"))
    parser.add_argument("--checkpoint", type=Path, default=BIDA_ROOT / "experiments/bida_memory_v1/formal/strict_2100/final.pth")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/bn_diagnostic/h1_seed2100")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(2100)
    np.random.seed(2100)
    device = torch.device(args.device)
    source_set, classes = make_dataset("Houston13", args.dataset_dir, source_train=True)
    target_set, target_classes = make_dataset("Houston18", args.dataset_dir, source_train=False)
    if target_classes != classes:
        raise ValueError("Source and target class counts differ")
    source_loader = make_loader(source_set, args.batch_size, args.workers)
    target_loader = make_loader(target_set, args.batch_size, args.workers)

    config = {
        "hypothesis": "H1 target-conditioned BatchNorm",
        "seed": 2100,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": 200,
        "dataset_dir": str(args.dataset_dir),
        "source_recalibration_pixels": len(source_set),
        "target_recalibration_pixels": len(target_set),
        "batch_size": args.batch_size,
        "recalibration": "reset buffers, cumulative statistics, one _tokenize pass, BN train only",
        "inference": "target self branch",
        "target_metric_timing": "prediction artifacts frozen before labels are collected",
        "protocol": str(Path(__file__).with_name("PROTOCOL.md")),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    variants = {}
    for name, domain_loader in (("original_mixed", None), ("source_bn", source_loader),
                                ("target_bn", target_loader)):
        model = build_model(args.checkpoint, device)
        recalibration_count = 0 if domain_loader is None else recalibrate(model, domain_loader, device)
        logits, predictions = predict(model, target_loader, device)
        artifact = args.out / f"predictions_{name}.npz"
        np.savez_compressed(artifact, logits=logits, predictions=predictions)
        buffers = args.out / f"bn_buffers_{name}.json"
        buffers.write_text(json.dumps(bn_snapshot(model)) + "\n")
        variants[name] = {
            "recalibration_count": recalibration_count,
            "prediction_artifact": str(artifact),
            "prediction_sha256": sha256(artifact),
            "bn_buffer_artifact": str(buffers),
            "bn_buffer_sha256": sha256(buffers),
        }
        print(json.dumps({"frozen": name, **variants[name]}), flush=True)
        del model

    manifest = {"config": config, "variants": variants}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = sha256(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_hash}), flush=True)

    # Post-hoc phase begins only after every prediction and its hash are frozen.
    labels = target_labels(target_loader)
    reports = {}
    predictions = {}
    for name, item in variants.items():
        frozen = np.load(item["prediction_artifact"])
        predictions[name] = frozen["predictions"]
        reports[name] = metrics(labels, predictions[name], frozen["logits"], classes)
    for left in variants:
        reports[left]["prediction_disagreement_percent"] = {
            right: float((predictions[left] != predictions[right]).mean() * 100)
            for right in variants if right != left
        }

    source = reports["source_bn"]
    target = reports["target_bn"]
    delta_oa = target["oa_percent"] - source["oa_percent"]
    delta_aa = target["aa_percent"] - source["aa_percent"]
    nonzero_recall = sum(value > 0 for value in target["per_class_percent"])
    largest_share_increase = (max(target["prediction_shares"]) -
                              max(source["prediction_shares"])) * 100
    checks = {
        "oa_gain_at_least_2pp": delta_oa >= 2.0,
        "aa_not_lower": delta_aa >= 0.0,
        "at_least_5_nonzero_recall_classes": nonzero_recall >= 5,
        "largest_prediction_share_increase_at_most_10pp": largest_share_increase <= 10.0,
    }
    result = {
        "frozen_manifest_sha256": manifest_hash,
        "reports": reports,
        "gate": {
            "target_bn_minus_source_bn_oa_points": delta_oa,
            "target_bn_minus_source_bn_aa_points": delta_aa,
            "target_bn_nonzero_recall_classes": nonzero_recall,
            "largest_prediction_share_increase_points": largest_share_increase,
            "checks": checks,
            "pass": all(checks.values()),
        },
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
