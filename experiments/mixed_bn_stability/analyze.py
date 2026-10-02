"""Audit mixed-BN seed instability using frozen models and trajectories."""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BIDA_ROOT = Path("/home/zhangzj26/IEEE_TCSVT_BiDA")
sys.path.insert(0, str(BIDA_ROOT))
sys.path.insert(0, str(BIDA_ROOT / "experiments/bida_memory_v1"))
from experiments.bida_memory_v1.run import make_loaders  # noqa: E402
from models.get_model import get_model  # noqa: E402
from utils.utils_HSI import seed_worker  # noqa: E402

helper_spec = importlib.util.spec_from_file_location(
    "dropout_helpers", ROOT / "experiments/bida_self_dropout/train.py"
)
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
hybrid_spec = importlib.util.spec_from_file_location(
    "hybrid_train", ROOT / "experiments/bida_self_tokenizer/train.py"
)
hybrid = importlib.util.module_from_spec(hybrid_spec)
hybrid_spec.loader.exec_module(hybrid)

SEEDS = (2100, 2101, 2102)
CONTROL_DIRS = {
    2100: ROOT / "results/bida_self/diagnostic_2100",
    2101: ROOT / "results/bida_self/multiseed_2101",
    2102: ROOT / "results/bida_self/multiseed_2102",
}
HYBRID_DIRS = {
    2100: ROOT / "results/bida_self_tokenizer/diagnostic_2100",
    2101: ROOT / "results/bida_self_tokenizer/formal_2101",
    2102: ROOT / "results/bida_self_tokenizer/formal_2102",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trainable_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not any(part in name for part in ("running_mean", "running_var", "num_batches_tracked")):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_model(arm, seed, device):
    opts = SimpleNamespace(num_tokens=4, dim=64, depth=3)
    model = get_model("BiDA", "Houston13", 13, opts)
    if arm == "hybrid":
        hybrid.transplant(model)
        checkpoint = HYBRID_DIRS[seed] / "checkpoints/epoch_200.pth"
    else:
        checkpoint = CONTROL_DIRS[seed] / "final.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["epoch"] != 200:
        raise RuntimeError(f"Expected epoch 200: {checkpoint}")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), checkpoint


def bn_snapshot(model):
    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            snapshot[name] = {
                "mean": module.running_mean.detach().cpu().numpy().tolist(),
                "var": module.running_var.detach().cpu().numpy().tolist(),
                "batches": int(module.num_batches_tracked),
            }
    return snapshot


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
        model._tokenize(images.to(device))
        count += len(images)
    model.eval()
    return count


@torch.inference_mode()
def predict_features(model, loader, device):
    logits, predictions, features = [], [], []
    model.eval()
    for images, _ in loader:
        images = images.to(device)
        output = model(images, images, return_feat_prob=True)
        batch_logits, batch_features = output[1], output[3]
        logits.append(batch_logits.cpu().numpy())
        predictions.append(batch_logits.argmax(1).cpu().numpy())
        features.append(batch_features.cpu().numpy())
    return (np.concatenate(logits).astype(np.float32),
            np.concatenate(predictions).astype(np.int64),
            np.concatenate(features).astype(np.float32))


def pairwise_bn_distances(snapshots):
    output = {}
    for layer in next(iter(snapshots.values())):
        output[layer] = {}
        for left in SEEDS:
            for right in SEEDS:
                if left >= right:
                    continue
                lm = np.asarray(snapshots[left][layer]["mean"])
                rm = np.asarray(snapshots[right][layer]["mean"])
                lv = np.asarray(snapshots[left][layer]["var"])
                rv = np.asarray(snapshots[right][layer]["var"])
                pooled_scale = np.sqrt(np.maximum((lv + rv) / 2, 1e-12))
                output[layer][f"{left}_{right}"] = {
                    "mean_l2": float(np.linalg.norm(lm - rm)),
                    "mean_standardized_rms": float(np.sqrt(np.mean(((lm - rm) / pooled_scale) ** 2))),
                    "log_variance_rms": float(np.sqrt(np.mean((np.log(lv) - np.log(rv)) ** 2))),
                }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="4")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-dir", default="/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.out / "artifacts"
    artifact_dir.mkdir()
    device = torch.device("cuda:" + args.device)

    opts = SimpleNamespace(seed=2100, num_workers=args.num_workers, dataset_dir=args.dataset_dir,
                           model="BiDA", lr=0.01, num_tokens=4, dim=64, depth=3,
                           loss_type="softmax", labelsmooth="off")
    seed_worker(2100)
    _, _, _, target_loader, source_loader, classes = make_loaders(opts)
    config = {"seeds": list(SEEDS), "arms": ["control", "hybrid"],
              "source_recalibration_pixels": len(source_loader.dataset),
              "target_recalibration_pixels": len(target_loader.dataset),
              "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md"))}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    records, original_snapshots = {}, {"control": {}, "hybrid": {}}
    for arm in ("control", "hybrid"):
        for seed in SEEDS:
            base, checkpoint = build_model(arm, seed, device)
            initial_trainable_hash = trainable_hash(base)
            original_snapshots[arm][seed] = bn_snapshot(base)
            for variant, loader in (("original_mixed", None),
                                    ("source_recalibrated", source_loader),
                                    ("target_recalibrated", target_loader)):
                model = copy.deepcopy(base)
                count = 0 if loader is None else recalibrate(model, loader, device)
                if trainable_hash(model) != initial_trainable_hash:
                    raise RuntimeError("A non-BN-buffer tensor changed during recalibration")
                logits, predictions, features = predict_features(model, target_loader, device)
                path = artifact_dir / f"{arm}_{seed}_{variant}.npz"
                np.savez_compressed(path, logits=logits, predictions=predictions, features=features)
                key = f"{arm}_{seed}_{variant}"
                records[key] = {"arm": arm, "seed": seed, "variant": variant,
                                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                                "trainable_sha256": initial_trainable_hash,
                                "recalibration_count": count, "artifact": str(path),
                                "artifact_sha256": sha256(path), "bn": bn_snapshot(model)}
                print(json.dumps({"frozen": key, "sha256": records[key]["artifact_sha256"]}), flush=True)
                del model
            del base

    trajectory = {}
    for seed, folder in HYBRID_DIRS.items():
        result = json.loads((folder / "results.json").read_text())
        trajectory[str(seed)] = [{"epoch": item["epoch"],
                                  "prediction_shares": item["metrics"]["prediction_shares"],
                                  "oa_percent": item["metrics"]["oa_percent"],
                                  "aa_percent": item["metrics"]["aa_percent"],
                                  "entropy": item["metrics"]["mean_entropy"]}
                                 for item in result["target_trajectory"]]
    manifest = {"config": config, "records": records,
                "original_bn_pairwise": {arm: pairwise_bn_distances(original_snapshots[arm])
                                         for arm in original_snapshots},
                "hybrid_frozen_trajectory": trajectory}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = sha256(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_hash}), flush=True)

    labels = np.concatenate([labels.numpy() for _, labels in target_loader])
    reports = {}
    for key, record in records.items():
        artifact = np.load(record["artifact"])
        report = helper.metrics(labels, artifact["predictions"], artifact["logits"], classes)
        features = artifact["features"]
        report["feature_norm_mean"] = float(np.linalg.norm(features, axis=1).mean())
        report["feature_channel_mean_rms"] = float(np.sqrt(np.mean(features.mean(0) ** 2)))
        reports[key] = report
    result = {"frozen_manifest_sha256": manifest_hash, "reports": reports,
              "original_bn_pairwise": manifest["original_bn_pairwise"],
              "hybrid_frozen_trajectory": trajectory}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({key: {"oa": value["oa_percent"], "aa": value["aa_percent"],
                                              "class6_share": value["prediction_shares"][5]}
                                  for key, value in reports.items()}), flush=True)


if __name__ == "__main__":
    main()
