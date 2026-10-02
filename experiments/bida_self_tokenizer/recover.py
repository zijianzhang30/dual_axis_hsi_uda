"""Complete post-training trajectory freezing after an interrupted prediction write."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tokenizer_train", Path(__file__).with_name("train.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2100, 2101, 2102), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.out / "config.json").read_text())
    if (args.out / "results.json").exists() or (args.out / "frozen_manifest.json").exists():
        parser.error("Run is already complete")

    opts = SimpleNamespace(seed=args.seed, num_workers=4,
                           dataset_dir="/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/",
                           model="BiDA", lr=0.01, num_tokens=4, dim=64, depth=3,
                           loss_type="softmax", labelsmooth="off")
    run.seed_worker(args.seed)
    _, _, _, target_eval_loader, _, classes = run.make_loaders(opts)
    model = run.get_model("BiDA", "Houston13", 13, opts)
    unused_ema = run.get_model("BiDA", "Houston13", 13, opts, ema=True)
    del unused_ema
    run.transplant(model)
    device = torch.device("cuda:" + args.device)
    model = model.to(device)

    checkpoint_dir = args.out / "checkpoints"
    prediction_dir = args.out / "predictions"
    checkpoints, predictions = [], []
    for epoch in range(10, 201, 10):
        checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        checkpoints.append({"epoch": epoch, "path": str(checkpoint),
                            "sha256": run.helper.file_hash(checkpoint)})
        prediction = prediction_dir / f"epoch_{epoch:03d}.npz"
        if epoch == 200:
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            logits, predicted = run.helper.predict(model, target_eval_loader, device)
            np.savez_compressed(prediction, logits=logits, predictions=predicted)
        artifact = np.load(prediction)
        if set(artifact.files) != {"logits", "predictions"} or len(artifact["predictions"]) != 52901:
            raise RuntimeError(f"Invalid prediction artifact: {prediction}")
        predictions.append({"epoch": epoch, "path": str(prediction),
                            "sha256": run.helper.file_hash(prediction)})

    manifest = {"config": config, "checkpoints": checkpoints, "predictions": predictions,
                "train_seconds": None, "trajectory_inference_seconds": None,
                "recovery": "epoch-200 prediction regenerated from frozen checkpoint after ENOSPC"}
    manifest_path = args.out / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = run.helper.file_hash(manifest_path)

    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in target_eval_loader])
    trajectory = []
    for item in predictions:
        artifact = np.load(item["path"])
        trajectory.append({"epoch": item["epoch"],
                           "metrics": run.helper.metrics(labels, artifact["predictions"],
                                                         artifact["logits"], classes)})
    final = trajectory[-1]["metrics"]
    best = max(trajectory, key=lambda row: row["metrics"]["oa_percent"])
    control = json.loads(run.control_result_path(args.seed).read_text())["target_at_final"]
    result = {"frozen_manifest_sha256": manifest_hash, "target_at_final": final,
              "target_oracle": {"best_epoch": best["epoch"], "metrics": best["metrics"]},
              "target_trajectory": trajectory, "bida_tokenizer_control_final": control,
              "transplant_minus_control_final": {"oa_points": final["oa_percent"] - control["oa_percent"],
                                                   "aa_points": final["aa_percent"] - control["aa_percent"],
                                                   "kappa": final["kappa"] - control["kappa"]},
              "recovery": manifest["recovery"]}
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps({"target_at_final": final, "target_oracle": result["target_oracle"],
                                  "transplant_minus_control_final": result["transplant_minus_control_final"],
                                  "frozen_manifest_sha256": manifest_hash}), flush=True)


if __name__ == "__main__":
    main()
