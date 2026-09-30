"""Paired A1-BN diagnostic with identical source parameter trajectory."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments/v0"))
from data import DEFAULT_BIDA_ROOT, DEFAULT_DATA_DIR, source_loaders, target_loader, target_train_loader  # noqa: E402
from model import DualAxisModel  # noqa: E402


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(bands, classes, patch_size, device):
    model = DualAxisModel(bands=bands, classes=classes, patch_size=patch_size, variant="a1")
    for index in (1, 4, 7):
        groups = model.stem[index]
        assert isinstance(groups, torch.nn.GroupNorm)
        model.stem[index] = torch.nn.BatchNorm3d(groups.num_channels)
    return model.to(device)


def capture_bn(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            result[name] = {
                "running_mean": module.running_mean.detach().clone(),
                "running_var": module.running_var.detach().clone(),
                "num_batches_tracked": module.num_batches_tracked.detach().clone(),
            }
    return result


def restore_bn(model, values):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in values:
                for key, value in values[name].items():
                    getattr(module, key).copy_(value)


def snapshot_state(model, buffers):
    restore_bn(model, buffers)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    logits = []
    for images, _ in loader:
        output = model(images.to(device))["logits_self"]
        logits.append(output.detach().cpu().numpy())
    logits = np.concatenate(logits).astype(np.float32, copy=False)
    return logits, logits.argmax(1).astype(np.int64, copy=False)


@torch.inference_mode()
def source_validation(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    for images, labels in loader:
        predictions = model(images.to(device))["logits_self"].argmax(1).cpu()
        correct += int((predictions == labels).sum())
        count += len(labels)
    return 100.0 * correct / count


def target_metrics(labels, predictions, logits, classes):
    cm = confusion_matrix(labels, predictions, labels=np.arange(classes))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(classes, dtype=float), where=cm.sum(1) != 0)
    shifted = logits - logits.max(1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(1, keepdims=True)
    counts = np.bincount(predictions, minlength=classes)
    return {
        "n": int(len(labels)),
        "oa_percent": float(np.trace(cm) / cm.sum() * 100),
        "aa_percent": float(recall.mean() * 100),
        "kappa": float(cohen_kappa_score(labels, predictions, labels=np.arange(classes))),
        "per_class_percent": (recall * 100).tolist(),
        "prediction_counts": counts.tolist(),
        "prediction_shares": (counts / counts.sum()).tolist(),
        "mean_entropy": float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1).mean()),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bida-root", type=Path, default=DEFAULT_BIDA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(args.device)
    source, source_val, classes, bands = source_loaders(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size, args.workers, 13)
    target_train = target_train_loader(
        args.dataset_dir, args.bida_root, args.seed, args.batch_size, args.workers, 13,
        generator=source.generator)
    model = make_model(bands, classes, 13, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_bn = capture_bn(model)
    n0_bn = {n: {k: v.clone() for k, v in state.items()} for n, state in initial_bn.items()}
    n1_bn = {n: {k: v.clone() for k, v in state.items()} for n, state in initial_bn.items()}
    config = {
        "arms": ["n0_source_bn", "n1_source_target_bn"],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patch_size": 13,
        "optimizer": "SGD lr=0.01, no momentum or schedule",
        "objective": "source CE only",
        "checkpoint_selection": f"fixed_epoch_{args.epochs}",
        "source_train_pixels": len(source.dataset),
        "source_val_pixels": len(source_val.dataset),
        "target_train_pixels": len(target_train.dataset),
        "target_gt_use": "GT mask selects unlabeled target centers; labels excluded from training and selection",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_parameter_sha256": parameter_hash(model),
        "protocol_sha256": digest_file(Path(__file__).with_name("PROTOCOL.md")),
        "dataset_dir": str(args.dataset_dir),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.monotonic()
    maximum_source_logit_difference = 0.0
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            source_ce_sum = 0.0
            count = 0
            steps = 0
            for (source_images, labels), (target_images, _) in zip(source, target_train):
                if len(source_images) != len(target_images):
                    continue
                source_images = source_images.to(device)
                labels = labels.to(device)
                target_images = target_images.to(device)

                restore_bn(model, n1_bn)
                with torch.no_grad():
                    n1_source_logits = model(source_images)["logits_self"]
                n1_bn = capture_bn(model)

                restore_bn(model, n0_bn)
                optimizer.zero_grad(set_to_none=True)
                n0_source_logits = model(source_images)["logits_self"]
                difference = float((n0_source_logits.detach() - n1_source_logits).abs().max())
                maximum_source_logit_difference = max(maximum_source_logit_difference, difference)
                if difference > 1e-4:
                    raise RuntimeError(f"Source logits diverged at epoch {epoch}, step {steps}: {difference}")
                n0_bn = capture_bn(model)
                ce = F.cross_entropy(n0_source_logits, labels)
                ce.backward()
                optimizer.step()

                restore_bn(model, n1_bn)
                with torch.no_grad():
                    model(target_images)
                n1_bn = capture_bn(model)
                source_ce_sum += float(ce.detach()) * len(labels)
                count += len(labels)
                steps += 1
            if steps == 0:
                raise RuntimeError("No paired steps")
            record = {"epoch": epoch, "steps": steps, "source_ce": source_ce_sum / count,
                      "max_source_logit_difference": maximum_source_logit_difference,
                      "parameter_sha256": parameter_hash(model)}
            if epoch == 1 or epoch % 10 == 0:
                model.eval()
                restore_bn(model, n0_bn)
                record["source_val_n0_oa_percent"] = source_validation(model, source_val, device)
                restore_bn(model, n1_bn)
                record["source_val_n1_oa_percent"] = source_validation(model, source_val, device)
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(record), flush=True)

    train_seconds = time.monotonic() - start
    parameter_sha256 = parameter_hash(model)
    target_eval, target_classes = target_loader(args.dataset_dir, args.bida_root, args.seed,
                                                 args.batch_size, args.workers, 13)
    if target_classes != classes:
        raise RuntimeError("Class counts differ")
    frozen = {}
    for arm, buffers in (("n0_source_bn", n0_bn), ("n1_source_target_bn", n1_bn)):
        state = snapshot_state(model, buffers)
        checkpoint = args.out / f"{arm}_final.pth"
        torch.save({"model": state, "epoch": args.epochs, "config": config}, checkpoint)
        model.eval()
        infer_start = time.monotonic()
        logits, predictions = predict(model, target_eval, device)
        infer_seconds = time.monotonic() - infer_start
        prediction_file = args.out / f"{arm}_predictions.npz"
        np.savez_compressed(prediction_file, logits=logits, predictions=predictions)
        frozen[arm] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
            "prediction_file": str(prediction_file), "prediction_sha256": digest_file(prediction_file),
            "inference_seconds": infer_seconds,
        }
        print(json.dumps({"frozen": arm, **frozen[arm]}), flush=True)

    manifest = {"config": config, "parameter_sha256_both_arms": parameter_sha256,
                "source_logit_max_abs_diff": maximum_source_logit_difference,
                "train_seconds_shared": train_seconds, "frozen": frozen}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_sha256 = digest_file(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_sha256}), flush=True)

    # All target prediction artifacts are frozen before labels are collected.
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in target_eval])
    reports = {}
    arm_predictions = {}
    for arm in frozen:
        artifact = np.load(frozen[arm]["prediction_file"])
        arm_predictions[arm] = artifact["predictions"]
        reports[arm] = target_metrics(labels, artifact["predictions"], artifact["logits"], classes)
    n0 = reports["n0_source_bn"]
    n1 = reports["n1_source_target_bn"]
    checks = {
        "oa_gain_at_least_2pp": n1["oa_percent"] - n0["oa_percent"] >= 2,
        "aa_not_lower": n1["aa_percent"] >= n0["aa_percent"],
        "at_least_5_nonzero_recall_classes": sum(x > 0 for x in n1["per_class_percent"]) >= 5,
        "largest_prediction_share_increase_at_most_10pp":
            max(n1["prediction_shares"]) - max(n0["prediction_shares"]) <= 0.10,
    }
    result = {"frozen_manifest_sha256": manifest_sha256, "reports": reports,
              "gate": {"checks": checks, "pass": all(checks.values()),
                       "n1_minus_n0_oa_points": n1["oa_percent"] - n0["oa_percent"],
                       "n1_minus_n0_aa_points": n1["aa_percent"] - n0["aa_percent"],
                       "prediction_disagreement_percent": float((arm_predictions["n0_source_bn"] !=
                                                                    arm_predictions["n1_source_target_bn"]).mean() * 100)},
              "shared_parameter_sha256": parameter_sha256,
              "maximum_source_logit_difference": maximum_source_logit_difference,
              "train_seconds_shared": train_seconds,
              "inference_seconds": {arm: item["inference_seconds"] for arm, item in frozen.items()},
              "parameter_count": config["parameter_count"]}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
