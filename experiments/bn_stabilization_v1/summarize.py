"""Verify frozen artifacts and report predeclared stability gates."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2100, 2101, 2102)
ARMS = ("original_mixed", "dsbn", "fixed_mixture")
OUT = ROOT / "results/bn_stabilization_v1"
LABELS = {"original_mixed": "Original Mixed BN", "dsbn": "DSBN", "fixed_mixture": "Fixed-Mixture 0.5/0.5"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    runs, summaries = {}, {}
    checked = 0
    for arm in ARMS:
        metrics = []
        for seed in SEEDS:
            if arm == "original_mixed":
                folder = ROOT / "results/bida_self" / ("diagnostic_2100" if seed == 2100 else f"multiseed_{seed}")
            else:
                folder = OUT / f"{arm}_{seed}"
            result = json.loads((folder / "results.json").read_text())
            manifest = json.loads((folder / "frozen_manifest.json").read_text())
            assert sha256(folder / "frozen_manifest.json") == result["frozen_manifest_sha256"]
            if arm != "original_mixed":
                for group in ("checkpoints", "predictions"):
                    assert [item["epoch"] for item in manifest[group]] == list(range(10, 201, 10))
                    for item in manifest[group]:
                        assert sha256(item["path"]) == item["sha256"]
                        checked += 1
            history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
            assert len(history) == 200 and history[-1]["epoch"] == 200
            final = dict(result["target_at_final"])
            final["class6_collapse"] = final["prediction_shares"][5] >= 0.95
            final["nonzero_recall_classes"] = sum(value > 0 for value in final["per_class_percent"])
            runs[f"{arm}_{seed}"] = {"folder": str(folder), "final": final,
                                    "source_val_final": history[-1]["source_val"],
                                    "manifest_sha256": result["frozen_manifest_sha256"],
                                    "trajectory": result.get("target_trajectory"),
                                    "learned_tensor_comparison": result.get("control_learned_tensor_comparison"),
                                    "max_source_ce_replay_error": result.get("max_source_ce_replay_error")}
            metrics.append(final)
        summary = {"collapse_count": sum(row["class6_collapse"] for row in metrics)}
        for metric in ("oa_percent", "aa_percent", "kappa"):
            values = np.asarray([row[metric] for row in metrics])
            summary[metric] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
        recalls = np.asarray([row["per_class_percent"] for row in metrics])
        summary["per_class_mean"] = recalls.mean(0).tolist()
        summary["per_class_sample_sd"] = recalls.std(0, ddof=1).tolist()
        summaries[arm] = summary

    gates = {}
    original = summaries["original_mixed"]
    for arm in ARMS[1:]:
        summary = summaries[arm]
        successful_seed_change = np.mean([runs[f"{arm}_{seed}"]["final"]["oa_percent"] -
                                         runs[f"original_mixed_{seed}"]["final"]["oa_percent"]
                                         for seed in (2100, 2102)])
        gate = {"no_epoch200_class6_collapse": summary["collapse_count"] == 0,
                "mean_oa_not_lower": summary["oa_percent"]["mean"] >= original["oa_percent"]["mean"],
                "mean_aa_not_lower": summary["aa_percent"]["mean"] >= original["aa_percent"]["mean"],
                "oa_sample_sd_lower": summary["oa_percent"]["sample_sd"] < original["oa_percent"]["sample_sd"],
                "successful_control_seeds_mean_oa_drop_at_most_2pp": bool(successful_seed_change >= -2.0)}
        gates[arm] = {"checks": gate, "pass": all(gate.values()),
                      "successful_control_seeds_mean_oa_change": float(successful_seed_change)}

    payload = {"summaries": summaries, "gates": gates, "runs": runs,
               "verified_new_checkpoint_prediction_artifacts": checked}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# BN stabilization V1: three-seed result", "",
             "Houston13 to Houston18, original BiDA tokenizer, depth 3, dropout 0.1,",
             "source self CE, fixed epoch 200; seeds 2100/2101/2102. Collapse means",
             "at least 95% of target predictions are class 6. Mean plus sample SD.", "",
             "| BN arm | Collapsed seeds | OA | AA | Kappa |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = summaries[arm]
        cells = [f"{row[key]['mean']:.2f} ± {row[key]['sample_sd']:.2f}" for key in ("oa_percent", "aa_percent")]
        k = row["kappa"]
        lines.append(f"| {LABELS[arm]} | {row['collapse_count']}/3 | {cells[0]} | {cells[1]} | {k['mean']:.3f} ± {k['sample_sd']:.3f} |")
    fixed_gate = gates["fixed_mixture"]
    lines += ["", f"Fixed-Mixture passes the preregistered confirmation gate: {fixed_gate['pass']}.",
              "It preserves the previously successful seeds and repairs seed 2101. The",
              "three-seed result supports a controlled follow-up of shared mixed training",
              "normalization. DSBN restores balanced recall but does not preserve OA.", "",
              "The intervention does not isolate update order: Fixed-Mixture also uses",
              "pooled current moments for both domains, includes between-domain mean",
              "variance, and allows target gradients through those moments. Its EMA rate",
              "matches the original per-step forgetting rate. This is evidence for the",
              "complete normalization protocol, not attribution to a single component.", "",
              "Minority-class instability remains: Fixed-Mixture class-1 and class-7 recall",
              "sample SDs are about 24.50 and 22.66 points. Three optimization seeds in one",
              "transfer pair do not establish stability across datasets or class priors."]
    lines += ["", "## Per-seed fixed endpoints", "",
              "| Arm | Seed | OA | AA | Class-6 share | Recall coverage | Source-val OA |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for seed in SEEDS:
            run = runs[f"{arm}_{seed}"]
            row = run["final"]
            lines.append(f"| {LABELS[arm]} | {seed} | {row['oa_percent']:.2f} | {row['aa_percent']:.2f} | "
                         f"{100*row['prediction_shares'][5]:.2f}% | {row['nonzero_recall_classes']}/7 | {run['source_val_final']['oa_percent']:.2f} |")
    lines += ["", "## Predeclared confirmation gate", "",
              "| Criterion | DSBN | Fixed-Mixture |", "| --- | --- | --- |"]
    for key in gates["dsbn"]["checks"]:
        lines.append(f"| {key} | {gates['dsbn']['checks'][key]} | {gates['fixed_mixture']['checks'][key]} |")
    for arm in ARMS[1:]:
        lines.append(f"\n{LABELS[arm]} passes all criteria: **{gates[arm]['pass']}**. "
                     f"Mean OA change on previously noncollapsed seeds 2100/2102: "
                     f"{gates[arm]['successful_control_seeds_mean_oa_change']:+.2f} points.")
    lines += ["", "## Per-class recall", "",
              "| Arm | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        row = summaries[arm]
        lines.append(f"| {LABELS[arm]} | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in zip(row["per_class_mean"], row["per_class_sample_sd"])) + " |")
    lines += ["", "## Frozen trajectory stability", "",
              "| Arm | Seed | Collapsed grid points | First collapsed epoch |", "| --- | ---: | ---: | ---: |"]
    for arm in ARMS[1:]:
        for seed in SEEDS:
            collapsed = [row["epoch"] for row in runs[f"{arm}_{seed}"]["trajectory"] if row["metrics"]["class6_collapse"]]
            lines.append(f"| {LABELS[arm]} | {seed} | {len(collapsed)}/20 | {collapsed[0] if collapsed else 'none'} |")
    lines += ["", "Avoiding class-6 collapse alone is not sufficient to establish overall",
              "trajectory stability. DSBN seed 2101 reaches only 12.88% OA at epoch 180,",
              "with 51.51% of predictions assigned to class 4; DSBN seed 2102 reaches",
              "21.25% OA at epoch 20. Fixed-Mixture has no comparable later breakdown",
              "in the frozen grid. These observations do not change the primary endpoint",
              "or predeclared gate."]
    lines += ["", "## Causal and reproducibility checks", ""]
    for seed in SEEDS:
        row = runs[f"dsbn_{seed}"]
        lines.append(f"- DSBN seed {seed}: final learned tensors exactly equal control = "
                     f"{row['learned_tensor_comparison']['exact']}; maximum epoch source-CE replay error "
                     f"= {row['max_source_ce_replay_error']:.3g}.")
    lines += ["", "DSBN changes only domain buffers. Fixed-Mixture uses shared current batch",
              "moments and changes gradients/learned weights. The fixed-mixture EMA rate",
              "is 0.19 per paired step, matching two original momentum-0.1 updates'",
              "forgetting rate. All arms have 376,567 trainable parameters.", "",
              f"Verified {checked} new checkpoint/prediction SHA256 records plus six manifests.",
              "Target predictions for all grid checkpoints were frozen before post-hoc",
              "target metrics. Original controls have only frozen epoch-200 endpoints.", "",
              "Artifacts and summary.json are under results/bn_stabilization_v1/.",
              "Trajectory figure: results/bn_stabilization_v1/trajectories.png."]
    Path(__file__).with_name("RESULTS.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    plot_fields = (("oa_percent", "Target OA (%)"), ("aa_percent", "Target AA (%)"),
                   ("prediction_shares", "Class-6 predictions (%)"))
    for row_index, seed in enumerate(SEEDS):
        for column, (field, label) in enumerate(plot_fields):
            axis = axes[row_index, column]
            for arm, color in (("dsbn", "tab:blue"), ("fixed_mixture", "tab:orange")):
                trajectory = runs[f"{arm}_{seed}"]["trajectory"]
                values = [100*item["metrics"][field][5] if field == "prediction_shares" else item["metrics"][field]
                          for item in trajectory]
                axis.plot([item["epoch"] for item in trajectory], values, label=LABELS[arm], color=color)
            original_final = runs[f"original_mixed_{seed}"]["final"]
            baseline_value = 100*original_final[field][5] if field == "prediction_shares" else original_final[field]
            axis.scatter([200], [baseline_value], marker="x", color="black", label="Original (epoch 200)", zorder=5)
            if field == "prediction_shares":
                axis.axhline(95, color="red", linestyle="--", linewidth=0.8, label="Collapse threshold")
            axis.set_title(f"Seed {seed}: {label}")
            axis.set_ylim(0, 103)
            axis.grid(alpha=0.25)
            if row_index == 2:
                axis.set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"summaries": summaries, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
