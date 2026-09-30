"""Aggregate paired A1 and A2-lite fixed-epoch Houston results."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("a1", "a2_lite")
SEEDS = (2100, 2101, 2102)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def load():
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = ROOT / "results" / "a2_lite" / f"{arm}_{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
            if len(history) != 200 or summary["config"]["checkpoint_selection"] != "fixed_epoch_200":
                raise ValueError(f"Incomplete or nonformal run: {folder}")
            runs[(arm, seed)] = {"config": summary["config"],
                                 "target": summary["target_at_final"],
                                 "source_val": history[-1]["source_val"],
                                 "final_train": history[-1]["losses"]}
    return runs


def main():
    runs = load()
    lines = ["# A1 vs A2-lite: explicit global spectral tokens", "",
             "Houston13 → Houston18, three optimization seeds, fixed epoch 200.",
             "Both arms use the same Strict BiDA data/training pipeline and source CE only.",
             "The released target GT mask defines eligible target centers; target labels",
             "enter neither loss nor checkpoint selection.", "",
             "| Arm | Target OA (%) | Target AA (%) | Kappa | Parameters |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        run_list = [runs[(arm, seed)] for seed in SEEDS]
        target = [run["target"] for run in run_list]
        params = {run["config"]["trainable_parameters"] for run in run_list}
        if len(params) != 1:
            raise ValueError(f"Parameter count differs across {arm} seeds")
        lines.append(f"| {arm} | {mean_sd([x['oa_percent'] for x in target])} | "
                     f"{mean_sd([x['aa_percent'] for x in target])} | "
                     f"{mean_sd([x['kappa'] for x in target])} | {params.pop():,} |")
    diffs = [runs[("a2_lite", seed)]["target"]["oa_percent"] -
             runs[("a1", seed)]["target"]["oa_percent"] for seed in SEEDS]
    lines += ["", f"Paired OA difference A2-lite − A1: **{mean_sd(diffs)} points**.", "",
              "## Per-seed outcome", "",
              "Spectral class-query mass is the summed attention weight over four spectral",
              "tokens; it is a usage diagnostic, not a causal attribution.", "",
              "| Seed | A1 OA | A2-lite OA | Difference | A1 AA | A2-lite AA | A1 source val OA | A2-lite source val OA | Spectral attention mass |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        a1 = runs[("a1", seed)]
        lite = runs[("a2_lite", seed)]
        a1_target, lite_target = a1["target"], lite["target"]
        lines.append(f"| {seed} | {a1_target['oa_percent']:.2f} | "
                     f"{lite_target['oa_percent']:.2f} | "
                     f"{lite_target['oa_percent'] - a1_target['oa_percent']:+.2f} | "
                     f"{a1_target['aa_percent']:.2f} | {lite_target['aa_percent']:.2f} | "
                     f"{a1['source_val']['oa_percent']:.2f} | "
                     f"{lite['source_val']['oa_percent']:.2f} | "
                     f"{lite_target['class_attention_spectral_mass']:.3f} |")
    lines += ["", "## Per-class target recall (%)", "",
              "Mean ± sample SD across seeds; classes follow the BiDA loader order.", "",
              "| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        values = np.asarray([runs[(arm, seed)]["target"]["per_class_percent"]
                             for seed in SEEDS])
        lines.append(f"| {arm} | " + " | ".join(
            mean_sd(values[:, class_id]) for class_id in range(values.shape[1])) + " |")
    lines += ["", "## Spectral attention over latent wavelength positions", "",
              "Rows are spectral query slots; columns are the 12 latent spectral positions.",
              "Entries are average target attention weights, rounded to three decimals.", ""]
    for seed in SEEDS:
        attention = runs[("a2_lite", seed)]["target"]["mean_spectral_attn"]
        lines += [f"### Seed {seed}", "",
                  "| Query | " + " | ".join(str(i + 1) for i in range(12)) + " |",
                  "| --- | " + " | ".join("---:" for _ in range(12)) + " |"]
        for query_id, row in enumerate(attention):
            lines.append(f"| {query_id + 1} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
        lines.append("")
    lines += ["The attention distribution alone cannot establish which spectral information",
              "caused a prediction. Per-run summaries retain full confusion matrices and",
              "mean spatial/spectral attention maps.", ""]
    output = ROOT / "experiments" / "a2_lite" / "RESULTS.md"
    output.write_text("\n".join(lines))
    (ROOT / "results" / "a2_lite" / "aggregate.json").write_text(
        json.dumps({f"{arm}_{seed}": runs[(arm, seed)]
                    for arm in ARMS for seed in SEEDS}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
