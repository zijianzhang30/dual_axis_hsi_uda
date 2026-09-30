"""Replay BiDA-self while preventing target BN buffer updates from persisting."""

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
REFERENCE_HISTORY = PROJECT_ROOT / "results/bida_self/diagnostic_2100/history.jsonl"
REFERENCE_RESULT = PROJECT_ROOT / "results/bida_self/diagnostic_2100/results.json"


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


class SourceBNRecorder:
    """Capture each BN buffer after its first (source) call in a paired forward."""

    def __init__(self, model):
        self.modules = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        self.calls = {}
        self.source_state = {}
        self.handles = [module.register_forward_hook(self._hook(name)) for name, module in self.modules]

    def _hook(self, name):
        def capture(module, _inputs, _output):
            self.calls[name] = self.calls.get(name, 0) + 1
            if self.calls[name] == 1:
                self.source_state[name] = {
                    "running_mean": module.running_mean.detach().clone(),
                    "running_var": module.running_var.detach().clone(),
                    "num_batches_tracked": module.num_batches_tracked.detach().clone(),
                }
        return capture

    def begin(self):
        self.calls.clear()
        self.source_state.clear()

    def restore_source(self):
        if set(self.source_state) != {name for name, _ in self.modules}:
            raise RuntimeError("Did not capture every BN layer during source forward")
        with torch.no_grad():
            for name, module in self.modules:
                for key, value in self.source_state[name].items():
                    getattr(module, key).copy_(value)

    def close(self):
        for handle in self.handles:
            handle.remove()


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
    args.model, args.lr, args.num_tokens, args.dim, args.depth = "BiDA", 0.01, 4, 64, 3
    args.loss_type, args.labelsmooth = "softmax", "off"
    device = torch.device("cuda:" + args.device)
    reference_history = [json.loads(line) for line in REFERENCE_HISTORY.read_text().splitlines()]
    mixed_result = json.loads(REFERENCE_RESULT.read_text())["target_at_final"]

    seed_worker(args.seed)
    train_loader, val_loader, target_train_loader, target_eval_loader, _, classes = make_loaders(args)
    model = get_model("BiDA", "Houston13", 13, args)
    initial_hash = state_hash(model)
    if initial_hash != EXPECTED_INITIAL_HASH:
        raise RuntimeError(f"Initial state mismatch: {initial_hash}")
    unused_ema = get_model("BiDA", "Houston13", 13, args, ema=True)
    del unused_ema
    model = model.to(device)
    optimizer, scheduler = load_scheduler("BiDA", model, args)
    recorder = SourceBNRecorder(model)
    config = {"arm": "bida_self_source_bn", "seed": args.seed, "epochs": args.epochs,
              "initial_model_sha256": initial_hash, "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "objective": "source self CE only", "bn_rule": "restore buffers captured after source tokenization",
              "reference_history": str(REFERENCE_HISTORY), "checkpoint_selection": "fixed_epoch_200",
              "protocol_sha256": file_hash(Path(__file__).with_name("PROTOCOL.md"))}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.monotonic()
    max_loss_difference = 0.0
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            count = 0
            steps = 0
            for (source_images, labels), (target_images, _) in zip(train_loader, target_train_loader):
                if len(source_images) != len(target_images):
                    continue
                source_images, labels = source_images.to(device), labels.to(device)
                target_images = target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                recorder.begin()
                source_logits, _, _, _ = model(source_images, target_images)
                loss = F.cross_entropy(source_logits, labels)
                loss.backward()
                # Restoring before backward changes BN buffers that autograd
                # version-checks. Restore immediately after backward instead;
                # the target statistics still never reach the next step.
                recorder.restore_source()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
                steps += 1
            if scheduler is not None:
                scheduler.step()
            source_ce = loss_sum / count
            difference = abs(source_ce - reference_history[epoch - 1]["source_ce"])
            max_loss_difference = max(max_loss_difference, difference)
            if difference > 1e-7:
                raise RuntimeError(f"Source CE replay diverged at epoch {epoch}: {difference}")
            record = {"epoch": epoch, "steps": steps, "source_ce": source_ce,
                      "reference_source_ce_abs_difference": difference}
            if epoch == 1 or epoch % 10 == 0:
                record["source_val"] = source_evaluate(model, val_loader, device)
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(record), flush=True)
    recorder.close()
    train_seconds = time.monotonic() - start

    checkpoint = args.out / "final.pth"
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, checkpoint)
    infer_start = time.monotonic()
    logits, predictions = predict(model, target_eval_loader, device)
    inference_seconds = time.monotonic() - infer_start
    prediction_file = args.out / "predictions.npz"
    np.savez_compressed(prediction_file, logits=logits, predictions=predictions)
    manifest = {"config": config, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
                "prediction_file": str(prediction_file), "prediction_sha256": file_hash(prediction_file),
                "maximum_epoch_source_ce_difference": max_loss_difference,
                "train_seconds": train_seconds, "inference_seconds": inference_seconds}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_sha256 = file_hash(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_sha256, **manifest}), flush=True)

    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in target_eval_loader])
    report = metrics(labels, predictions, logits, classes)
    result = {"frozen_manifest_sha256": manifest_sha256, "source_bn_target_at_final": report,
              "mixed_bn_reference": mixed_result,
              "mixed_minus_source_bn": {"oa_points": mixed_result["oa_percent"] - report["oa_percent"],
                                        "aa_points": mixed_result["aa_percent"] - report["aa_percent"],
                                        "kappa": mixed_result["kappa"] - report["kappa"]},
              "maximum_epoch_source_ce_difference": max_loss_difference,
              "train_seconds": train_seconds, "inference_seconds": inference_seconds}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
