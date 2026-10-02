"""Three-seed BN stabilization V1 with frozen ten-epoch predictions."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from normalization import canonical_state, install, set_domain

ROOT = Path(__file__).resolve().parents[2]
BIDA_ROOT = Path("/home/zhangzj26/IEEE_TCSVT_BiDA")
sys.path.insert(0, str(BIDA_ROOT))
sys.path.insert(0, str(BIDA_ROOT / "experiments/bida_memory_v1"))
from experiments.bida_memory_v1.run import make_loaders  # noqa: E402
from models.get_model import get_model  # noqa: E402
from utils.scheduler import load_scheduler  # noqa: E402
from utils.utils_HSI import seed_worker  # noqa: E402

spec = importlib.util.spec_from_file_location("bida_helpers", ROOT / "experiments/bida_self_dropout/train.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def control_path(seed):
    folder = "diagnostic_2100" if seed == 2100 else f"multiseed_{seed}"
    return ROOT / "results/bida_self" / folder


def learned_tensor_comparison(model, control):
    reference = torch.load(control / "final.pth", map_location="cpu", weights_only=True)["model"]
    maximum = 0.0
    exact = True
    for key, value in canonical_state(model).items():
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        value = value.detach().cpu()
        exact &= torch.equal(value, reference[key])
        maximum = max(maximum, float((value - reference[key]).abs().max()))
    return {"exact": bool(exact), "max_abs_difference": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("dsbn", "fixed_mixture"), required=True)
    parser.add_argument("--seed", type=int, choices=(2100, 2101, 2102), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-dir", default="/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output must be empty: {args.out}")
    args.model, args.lr, args.num_tokens, args.dim, args.depth = "BiDA", 0.01, 4, 64, 3
    args.loss_type, args.labelsmooth, args.epochs = "softmax", "off", 200
    device = torch.device("cuda:" + args.device)
    control = control_path(args.seed)
    reference_config = json.loads((control / "config.json").read_text())
    reference_history = [json.loads(line) for line in (control / "history.jsonl").read_text().splitlines()]

    seed_worker(args.seed)
    train_loader, val_loader, target_train_loader, target_eval_loader, _, classes = make_loaders(args)
    model = get_model("BiDA", "Houston13", 13, args)
    initial_hash = helper.state_hash(model)
    if initial_hash != reference_config["initial_model_sha256"]:
        raise RuntimeError(f"Paired control initial hash mismatch: {initial_hash}")
    unused_ema = get_model("BiDA", "Houston13", 13, args, ema=True)
    del unused_ema
    checked = install(model, args.arm)
    model = model.to(device)
    optimizer, scheduler = load_scheduler("BiDA", model, args)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir, prediction_dir = args.out / "checkpoints", args.out / "predictions"
    checkpoint_dir.mkdir()
    prediction_dir.mkdir()
    config = {"arm": args.arm, "seed": args.seed, "epochs": 200,
              "initial_bida_sha256": initial_hash, "initial_shared_tensors_checked": checked,
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "depth": 3, "dropout": 0.1, "objective": "source self CE",
              "primary": "fixed_epoch_200", "mixture_source_weight": 0.5 if args.arm == "fixed_mixture" else None,
              "bn_momentum_per_step": 0.19 if args.arm == "fixed_mixture" else 0.1,
              "control_directory": str(control), "protocol_sha256": helper.file_hash(Path(__file__).with_name("PROTOCOL.md")),
              "train_script_sha256": helper.file_hash(Path(__file__)),
              "normalization_script_sha256": helper.file_hash(Path(__file__).with_name("normalization.py")),
              "device": str(device)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.monotonic()
    checkpoints = []
    max_replay_error = 0.0
    with (args.out / "history.jsonl").open("w") as history:
        for epoch in range(1, 201):
            model.train()
            loss_sum, count, steps = 0.0, 0, 0
            for (source_images, labels), (target_images, _) in zip(train_loader, target_train_loader):
                if len(source_images) != len(target_images):
                    continue
                source_images, labels = source_images.to(device), labels.to(device)
                target_images = target_images.to(device)
                optimizer.zero_grad(set_to_none=True)
                source_logits = model(source_images, target_images)[0]
                loss = F.cross_entropy(source_logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
                steps += 1
            if scheduler is not None:
                scheduler.step()
            record = {"epoch": epoch, "steps": steps, "source_ce": loss_sum / count}
            if args.arm == "dsbn":
                error = abs(record["source_ce"] - reference_history[epoch - 1]["source_ce"])
                max_replay_error = max(max_replay_error, error)
                record["control_source_ce_abs_error"] = error
                if error > 1e-7:
                    raise RuntimeError(f"DSBN source training replay diverged at epoch {epoch}: {error}")
            if epoch == 1 or epoch % 10 == 0:
                set_domain(model, "source")
                record["source_val"] = helper.source_evaluate(model, val_loader, device)
                set_domain(model, "target")
                print(json.dumps(record), flush=True)
            if epoch % 10 == 0:
                path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, path)
                checkpoints.append({"epoch": epoch, "path": str(path), "sha256": helper.file_hash(path)})
            history.write(json.dumps(record) + "\n")
            history.flush()
    train_seconds = time.monotonic() - start
    comparison = learned_tensor_comparison(model, control)
    if args.arm == "dsbn" and not comparison["exact"]:
        raise RuntimeError(f"DSBN learned tensors differ: {comparison}")

    predictions = []
    inference_start = time.monotonic()
    set_domain(model, "target")
    for item in checkpoints:
        model.load_state_dict(torch.load(item["path"], map_location=device, weights_only=True)["model"])
        logits, predicted = helper.predict(model, target_eval_loader, device)
        path = prediction_dir / f"epoch_{item['epoch']:03d}.npz"
        np.savez_compressed(path, logits=logits, predictions=predicted)
        predictions.append({"epoch": item["epoch"], "path": str(path), "sha256": helper.file_hash(path)})
        print(json.dumps({"frozen_prediction_epoch": item["epoch"]}), flush=True)
    inference_seconds = time.monotonic() - inference_start
    manifest = {"config": config, "checkpoints": checkpoints, "predictions": predictions,
                "train_seconds": train_seconds, "inference_seconds": inference_seconds,
                "control_learned_tensor_comparison": comparison, "max_source_ce_replay_error": max_replay_error}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = helper.file_hash(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_hash}), flush=True)

    labels = np.concatenate([labels.numpy() for _, labels in target_eval_loader])
    trajectory = []
    for item in predictions:
        with np.load(item["path"]) as artifact:
            report = helper.metrics(labels, artifact["predictions"], artifact["logits"], classes)
        report["class6_collapse"] = report["prediction_shares"][5] >= 0.95
        trajectory.append({"epoch": item["epoch"], "metrics": report})
    result = {"frozen_manifest_sha256": manifest_hash, "target_at_final": trajectory[-1]["metrics"],
              "target_trajectory": trajectory, "control_learned_tensor_comparison": comparison,
              "max_source_ce_replay_error": max_replay_error, "train_seconds": train_seconds,
              "inference_seconds": inference_seconds}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result["target_at_final"]), flush=True)


if __name__ == "__main__":
    main()
