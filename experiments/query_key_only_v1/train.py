"""Key-only spatial PE: paired three-seed test, no architecture additions."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import MethodType
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/query_class_readout_v1"))
from pipeline import install_readout
from key_only import install_key_only

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIDA_ROOT = Path("/home/zhangzj26/IEEE_TCSVT_BiDA")
sys.path.insert(0, str(BIDA_ROOT))
sys.path.insert(0, str(BIDA_ROOT / "experiments/bida_memory_v1"))

from experiments.bida_memory_v1.run import make_loaders  # noqa: E402
from models.get_model import get_model  # noqa: E402
from utils.scheduler import load_scheduler  # noqa: E402
from utils.utils_HSI import seed_worker  # noqa: E402

helper_spec = importlib.util.spec_from_file_location(
    "bida_dropout_helpers", PROJECT_ROOT / "experiments/bida_self_dropout/train.py"
)
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
a1_spec = importlib.util.spec_from_file_location("a1_model_for_tokenizer", PROJECT_ROOT / "model.py")
a1 = importlib.util.module_from_spec(a1_spec)
a1_spec.loader.exec_module(a1)

joint_spec = importlib.util.spec_from_file_location("hybrid_v2_joint", PROJECT_ROOT / "experiments/bn_stabilization_v1/normalization.py")
joint = importlib.util.module_from_spec(joint_spec)
joint_spec.loader.exec_module(joint)

EXPECTED_HASH = "dadb50a683a2bab506df66c317cbea8d08aee955bab5281013dd4604d6fa40b5"


def control_result_path(seed):
    return PROJECT_ROOT / f"results/query_class_readout_v1/formal_{seed}/results.json"


def a1_semantic_tokens(self, features):
    batch, channels, height, width = features.shape
    pixels = features.flatten(2).transpose(1, 2)
    half = channels // 2
    row = a1.sinusoidal_positions(height, half, features.device, features.dtype)
    col = a1.sinusoidal_positions(width, channels - half, features.device, features.dtype)
    position = torch.cat((row[:, None].expand(-1, width, -1),
                          col[None].expand(height, -1, -1)), dim=-1)
    if getattr(self, "_spatial_pe_enabled", True):
        pixels = pixels + position.reshape(1, height * width, channels)
    tokens, _ = self.spatial_reader(self.spatial_queries[None].expand(batch, -1, -1), pixels)
    return tokens


def transplant(model):
    before = {key: value.clone() for key, value in model.state_dict().items()
              if not key.startswith("conv_a.")}
    model.conv_a = nn.Identity()
    model.spatial_queries = nn.Parameter(torch.randn(4, 64) * 0.02)
    model.spatial_reader = a1.CrossBlock(64, heads=4)
    model._forward_semantic_tokens = MethodType(a1_semantic_tokens, model)
    after = model.state_dict()
    if not all(torch.equal(value, after[key]) for key, value in before.items()):
        raise RuntimeError("Shared BiDA initial tensors changed during tokenizer transplant")
    return len(before), sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="6")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-dir", default="/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seed not in {2100, 2101, 2102} or args.epochs != 200:
        parser.error("Key-only contrast requires seed 2100/2101/2102 and 200 epochs")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output must be empty: {args.out}")
    args.model, args.lr, args.num_tokens, args.dim, args.depth = "BiDA", 0.01, 4, 64, 3
    args.loss_type, args.labelsmooth = "softmax", "off"
    device = torch.device("cuda:" + args.device)
    control_path = control_result_path(args.seed)
    if not control_path.exists():
        parser.error(f"Paired BiDA-self control is missing: {control_path}")
    control = json.loads(control_path.read_text())["target_at_final"]

    seed_worker(args.seed)
    train_loader, val_loader, target_train_loader, target_eval_loader, _, classes = make_loaders(args)
    model = get_model("BiDA", "Houston13", 13, args)
    initial_hash = helper.state_hash(model)
    base_folder = PROJECT_ROOT / "results/bida_self" / ("diagnostic_2100" if args.seed == 2100 else f"multiseed_{args.seed}")
    expected = json.loads((base_folder / "config.json").read_text())["initial_model_sha256"]
    if initial_hash != expected:
        raise RuntimeError(f"Initial state mismatch: {initial_hash}")
    unused_ema = get_model("BiDA", "Houston13", 13, args, ema=True)
    del unused_ema
    shared_tensors, parameters = transplant(model)
    hybrid_folder = PROJECT_ROOT / "results/bida_self_tokenizer" / ("diagnostic_2100" if args.seed == 2100 else f"formal_{args.seed}")
    hybrid_hash = helper.state_hash(model)
    expected_hybrid = json.loads((hybrid_folder / "config.json").read_text())["initial_transplant_sha256"]
    if hybrid_hash != expected_hybrid:
        raise RuntimeError(f"Existing Hybrid initial state mismatch: {hybrid_hash}")
    parameters = install_readout(model, a1)
    args.depth = 0
    pipeline_hash = helper.state_hash(model)
    baseline_config = json.loads((control_path.with_name("config.json")).read_text())
    assert pipeline_hash == baseline_config["initial_pipeline_sha256"]
    for key, path in (("joint_script_sha256", PROJECT_ROOT / "experiments/bn_stabilization_v1/normalization.py"),
                      ("pipeline_script_sha256", PROJECT_ROOT / "experiments/query_class_readout_v1/pipeline.py"),
                      ("a1_script_sha256", PROJECT_ROOT / "model.py")):
        assert helper.file_hash(path) == baseline_config[key], key
    # PE is parameter-free and consumes no random draws. Same tensors, dropout
    # shapes, loader setup and evaluation cadence preserve the batch RNG path.
    rng_before = torch.get_rng_state().clone()
    install_key_only(model, a1)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert helper.state_hash(model) == pipeline_hash
    joint.install(model, "fixed_mixture")
    assert helper.state_hash(model) == pipeline_hash
    model = model.to(device)
    optimizer, scheduler = load_scheduler("BiDA", model, args)

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out / "checkpoints"
    prediction_dir = args.out / "predictions"
    checkpoint_dir.mkdir()
    prediction_dir.mkdir()
    config = {"arm": "query_key_only_pe", "spatial_pe": "key_only",
              "key_value_rule": "K=shared_LN(X+PE), V=shared_LN(X)",
              "key_only_script_sha256": helper.file_hash(Path(__file__).with_name("key_only.py")),
              "paired_control": str(control_path), "batch_rng_path": "unchanged; PE has no random draws", "seed": args.seed, "epochs": args.epochs,
              "initial_bida_sha256": initial_hash, "shared_initial_tensors_verified": shared_tensors,
              "initial_full_transplant_sha256": hybrid_hash,
              "initial_pipeline_sha256": pipeline_hash,
              "trainable_parameters": parameters, "depth": 0, "dropout": 0.1,
              "bn": "Full Joint native BN", "bn_momentum_per_step": 0.19, "mixture_source_weight": 0.5, "objective": "source self CE only",
              "primary": "fixed_epoch_200", "secondary": "target-oracle max over frozen epochs 10:10:200",
              "protocol_sha256": helper.file_hash(Path(__file__).with_name("PROTOCOL.md")),
              "train_script_sha256": helper.file_hash(Path(__file__)),
              "joint_script_sha256": helper.file_hash(PROJECT_ROOT / "experiments/bn_stabilization_v1/normalization.py"),
              "pipeline_script_sha256": helper.file_hash(PROJECT_ROOT / "experiments/query_class_readout_v1/pipeline.py"),
              "a1_script_sha256": helper.file_hash(PROJECT_ROOT / "model.py"),
              "readout": "A1 class query CrossBlock -> retained LayerNorm/linear",
              "token_dropout": 0.1}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.monotonic()
    checkpoints = []
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
                record["source_val"] = helper.source_evaluate(model, val_loader, device)
            if epoch % 10 == 0:
                path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, path)
                checkpoints.append({"epoch": epoch, "path": str(path), "sha256": helper.file_hash(path)})
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps(record), flush=True)
    train_seconds = time.monotonic() - start

    predictions = []
    infer_start = time.monotonic()
    for item in checkpoints:
        saved = torch.load(item["path"], map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        logits, predicted = helper.predict(model, target_eval_loader, device)
        path = prediction_dir / f"epoch_{item['epoch']:03d}.npz"
        np.savez_compressed(path, logits=logits, predictions=predicted)
        predictions.append({"epoch": item["epoch"], "path": str(path), "sha256": helper.file_hash(path)})
        print(json.dumps({"frozen_prediction_epoch": item["epoch"],
                          "sha256": predictions[-1]["sha256"]}), flush=True)
    inference_seconds = time.monotonic() - infer_start
    manifest = {"config": config, "checkpoints": checkpoints, "predictions": predictions,
                "train_seconds": train_seconds, "trajectory_inference_seconds": inference_seconds}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = helper.file_hash(manifest_path)
    print(json.dumps({"frozen_manifest_sha256": manifest_hash}), flush=True)

    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in target_eval_loader])
    trajectory = []
    for item in predictions:
        artifact = np.load(item["path"])
        trajectory.append({"epoch": item["epoch"],
                           "metrics": helper.metrics(labels, artifact["predictions"], artifact["logits"], classes)})
    for row in trajectory:
        row["metrics"]["class6_collapse"] = row["metrics"]["prediction_shares"][5] >= 0.95
    final = trajectory[-1]["metrics"]
    best = max(trajectory, key=lambda row: row["metrics"]["oa_percent"])
    result = {"frozen_manifest_sha256": manifest_hash, "target_at_final": final,
              "target_oracle": {"best_epoch": best["epoch"], "metrics": best["metrics"]},
              "target_trajectory": trajectory, "pe_on_control_final": control,
              "key_only_minus_pe_on_final": {"oa_points": final["oa_percent"] - control["oa_percent"],
                                                   "aa_points": final["aa_percent"] - control["aa_percent"],
                                                   "kappa": final["kappa"] - control["kappa"]},
              "train_seconds": train_seconds, "trajectory_inference_seconds": inference_seconds}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({"target_at_final": final, "target_oracle": result["target_oracle"],
                                  "key_only_minus_pe_on_final": result["key_only_minus_pe_on_final"]}), flush=True)


if __name__ == "__main__":
    main()
