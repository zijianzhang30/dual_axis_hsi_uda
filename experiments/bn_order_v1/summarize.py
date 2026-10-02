"""Compare frozen original/reverse sequential BN with Joint-Batch V1."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/bn_order_v1"
SEEDS = (2100, 2101, 2102)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    rows, runs = [], {}
    for seed in SEEDS:
        original_dir = ROOT / "results/bida_self" / ("diagnostic_2100" if seed == 2100 else f"multiseed_{seed}")
        reverse_dir = OUT / f"reverse_{seed}"
        joint_dir = ROOT / "results/bn_stabilization_v1" / f"fixed_mixture_{seed}"
        reverse = json.loads((reverse_dir / "results.json").read_text())
        manifest = json.loads((reverse_dir / "frozen_manifest.json").read_text())
        assert sha256(reverse_dir / "frozen_manifest.json") == reverse["frozen_manifest_sha256"]
        assert reverse["control_learned_tensor_comparison"]["exact"]
        assert reverse["max_source_ce_replay_error"] == 0
        for group in ("checkpoints", "predictions"):
            assert [item["epoch"] for item in manifest[group]] == list(range(10, 201, 10))
            for item in manifest[group]:
                assert sha256(item["path"]) == item["sha256"]
        original = json.loads((original_dir / "results.json").read_text())
        joint = json.loads((joint_dir / "results.json").read_text())
        original_state = torch.load(original_dir / "final.pth", map_location="cpu", weights_only=True)["model"]
        reverse_state = torch.load(reverse_dir / "checkpoints/epoch_200.pth", map_location="cpu", weights_only=True)["model"]
        buffer_changes = {}
        for key, before in original_state.items():
            if key.endswith("running_mean"):
                variance = original_state[key.replace("running_mean", "running_var")]
                delta = (reverse_state[key] - before) / variance.sqrt()
                buffer_changes[key] = float(delta.square().mean().sqrt())
            elif key.endswith("running_var"):
                buffer_changes[key] = float((reverse_state[key].log() - before.log()).square().mean().sqrt())
        with np.load(original_dir / "predictions.npz") as art:
            original_prediction = art["predictions"]
        with np.load(reverse_dir / "predictions/epoch_200.npz") as art:
            reverse_prediction = art["predictions"]
        original_final, reverse_final, joint_final = (x["target_at_final"] for x in (original, reverse, joint))
        rows.append({"seed": seed, "original": original_final, "reverse": reverse_final, "joint": joint_final,
                     "oa_order_change": reverse_final["oa_percent"] - original_final["oa_percent"],
                     "aa_order_change": reverse_final["aa_percent"] - original_final["aa_percent"],
                     "order_prediction_disagreement_percent": float((original_prediction != reverse_prediction).mean() * 100),
                     "bn_buffer_changes": buffer_changes})
        runs[str(seed)] = {"frozen_manifest_sha256": reverse["frozen_manifest_sha256"],
                           "max_source_ce_replay_error": reverse["max_source_ce_replay_error"],
                           "learned_tensors_exact": True, "trajectory": reverse["target_trajectory"]}
    summaries = {}
    for arm in ("original", "reverse", "joint"):
        metrics = [row[arm] for row in rows]
        summaries[arm] = {"class6_collapse_count": sum(row["prediction_shares"][5] >= 0.95 for row in metrics)}
        for field in ("oa_percent", "aa_percent", "kappa"):
            values = np.asarray([row[field] for row in metrics])
            summaries[arm][field] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
    payload = {"summaries": summaries, "paired_rows": rows, "runs": runs,
               "joint_batch_equals_fixed_mixture_v1": True,
               "verified_new_artifacts": 120}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# BN update-order causal control", "",
             "Seeds 2100/2101/2102; original BiDA tokenizer, depth 3, dropout 0.1,",
             "source CE, fixed epoch 200. Collapse threshold: >=95% class-6 predictions.", "",
             "| Arm | OA mean ± sample SD | AA mean ± sample SD | Class-6 collapse |",
             "| --- | ---: | ---: | ---: |"]
    names = {"original": "Source then target", "reverse": "Target then source", "joint": "Joint-Batch / Fixed-Mixture V1"}
    for arm in names:
        x = summaries[arm]
        lines.append(f"| {names[arm]} | {x['oa_percent']['mean']:.2f} ± {x['oa_percent']['sample_sd']:.2f} | "
                     f"{x['aa_percent']['mean']:.2f} ± {x['aa_percent']['sample_sd']:.2f} | {x['class6_collapse_count']}/3 |")
    lines += ["", "| Seed | Original OA | Reverse OA | Order OA change | Order AA change | Original class-6 share | Reverse class-6 share | Joint OA |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['original']['oa_percent']:.2f} | {row['reverse']['oa_percent']:.2f} | "
                     f"{row['oa_order_change']:+.2f} | {row['aa_order_change']:+.2f} | "
                     f"{100*row['original']['prediction_shares'][5]:.2f}% | {100*row['reverse']['prediction_shares'][5]:.2f}% | "
                     f"{row['joint']['oa_percent']:.2f} |")
    lines += ["", "## Mechanistic interpretation", "",
              "Changing only running-buffer update order lowers OA in every seed by",
              "0.97, 1.42 and 1.17 points (mean -1.19), and lowers mean AA by 5.57",
              "points. It leaves the same seed-2101 class-6 collapse in place, increasing",
              "that class's prediction share from 95.93% to 98.09%. In seed 2102,",
              "class-7 recall falls from 0.73% to zero. The order effect is therefore real",
              "and affects minority coverage, but merely reversing order does not",
              "explain or reproduce Joint-Batch's stability improvement.", "",
              "This supports studying the full shared-training-statistics protocol.",
              "It does not prove that order bias is irrelevant or that target gradients",
              "alone drive the gain. Joint-Batch and Fixed-Mixture V1 are the same arm",
              "under equal batch sizes and matched EMA rate; their equivalence cannot",
              "separate training moments from buffer moments or target gradient paths.",
              "A future buffer-only balanced update preserving domain-specific training",
              "normalization, followed by a detached-target-moment shared-normalization",
              "control, would isolate these contributions. Neither is run or selected",
              "in this study."]
    lines += ["", "## What is isolated", "",
              "Reverse-order source CE matches every epoch of the original control exactly",
              "in all three seeds. All final non-BN-buffer tensors also match exactly.",
              "Target/source stem order changes; source labels, downstream branches and",
              "dropout consumption retain their original order. This comparison therefore",
              "isolates the effect of BN running-buffer update order on inference.", "",
              "Sequential source-then-target fresh coefficients are 0.09/0.10; reverse",
              "coefficients are 0.10/0.09, with old-state weight 0.81 in both cases.",
              "Fresh normalized domain weights are 47.37/52.63 versus 52.63/47.37.", "",
              "## Joint-Batch identity", "",
              "Fixed-Mixture V1 already calls native BN once on concatenated equal-sized",
              "domain activations in each stem layer with momentum 0.19. An independently",
              "implemented concatenated stem produces identical tokens (max error 0),",
              "source/target input gradients, parameter gradients and BN buffers on the",
              "verification fixture. It is the Joint-Batch arm under the same EMA rate.",
              "The existing three-seed frozen results are reused; there is no distinct",
              "special mixture implementation to compare against Joint-Batch.", "",
              "These comparisons cannot isolate pooled training normalization from",
              "the between-domain variance term or target moment gradients. A buffer-only",
              "balanced-mixture control or detached-target-moment training intervention",
              "would be needed to distinguish those mechanisms, with a protocol frozen",
              "before further target outcomes.", "",
              "## Frozen reverse trajectories", "",
              "| Seed | Class-6 collapsed grid points | First collapsed epoch |",
              "| --- | ---: | ---: |"]
    for seed in SEEDS:
        collapsed = [item["epoch"] for item in runs[str(seed)]["trajectory"] if item["metrics"]["class6_collapse"]]
        lines.append(f"| {seed} | {len(collapsed)}/20 | {collapsed[0] if collapsed else 'none'} |")
    lines += ["", "Verified all 120 new checkpoint/prediction SHA256 records and three",
              "frozen manifests. All twenty predictions per seed were frozen before target",
              "metrics. Full per-class recalls, prediction shares, buffer differences,",
              "and trajectories are in results/bn_order_v1/summary.json."]
    Path(__file__).with_name("RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summaries": summaries, "order_changes": [{key: row[key] for key in ("seed", "oa_order_change", "aa_order_change", "order_prediction_disagreement_percent", "bn_buffer_changes")} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
