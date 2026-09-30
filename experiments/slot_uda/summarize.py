"""Aggregate the locked B0-B4 matrix after every formal run has finished."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2100, 2101, 2102)
ARMS = ("b0", "b1", "b2", "b3", "b4")


def run_dir(arm, seed):
    folder = "v0" if arm == "b0" else "slot_uda"
    name = f"a1_{seed}" if arm == "b0" else f"{arm}_{seed}"
    return ROOT / "results" / folder / name


def aggregate():
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = run_dir(arm, seed)
            summary_file = path / "summary.json"
            if not summary_file.exists():
                raise FileNotFoundError(f"Missing completed run: {summary_file}")
            summary = json.loads(summary_file.read_text())
            if summary["config"]["checkpoint_selection"] != "fixed_epoch_200":
                raise ValueError(f"Unexpected checkpoint protocol: {summary_file}")
            history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
            if len(history) != 200:
                raise ValueError(f"Incomplete run: {path}")
            if arm == "b0":
                diagnostics = json.loads((ROOT / "results" / "slot_uda" /
                                          f"b0_diagnostics_{seed}.json").read_text())
                target = dict(summary["target_at_final"])
                target.update({key: diagnostics["target"][key]
                               for key in ("mean_slot_cosine", "mean_attention_cosine")})
                source_val = diagnostics["source_val"]
            else:
                target = summary["target_at_final"]
                source_val = summary["source_val_at_final"]
            runs[(arm, seed)] = {"target": target, "source_val": source_val,
                                 "first_epoch_losses": history[0]["losses"],
                                 "final_epoch_losses": history[-1]["losses"]}
    return runs


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def offdiag_mean(matrix):
    matrix = np.asarray(matrix)
    return float(matrix[np.triu_indices(len(matrix), 1)].mean())


def main():
    runs = aggregate()
    lines = ["# B0–B4 Houston13 → Houston18", "",
             "Primary endpoint: target self OA at fixed epoch 200; same three optimization seeds.",
             "Target GT mask supplies training candidate centers, but target labels enter no loss or checkpoint choice.",
             "", "| Arm | OA (%) | AA (%) | Kappa |", "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        records = [runs[(arm, seed)]["target"] for seed in SEEDS]
        lines.append(f"| {arm.upper()} | {mean_sd([x['oa_percent'] for x in records])} | "
                     f"{mean_sd([x['aa_percent'] for x in records])} | "
                     f"{mean_sd([x['kappa'] for x in records])} |")
    lines += ["", "| Paired difference | Target OA points, mean ± SD |",
              "| --- | ---: |"]
    for left, right in (("b2", "b1"), ("b4", "b2"), ("b3", "b0"), ("b4", "b3")):
        diffs = [runs[(left, seed)]["target"]["oa_percent"] -
                 runs[(right, seed)]["target"]["oa_percent"] for seed in SEEDS]
        lines.append(f"| {left.upper()} − {right.upper()} | {mean_sd(diffs)} |")
    lines += ["", "## Per-seed OA (%)", "",
              "| Seed | B0 | B1 | B2 | B3 | B4 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        values = [runs[(arm, seed)]["target"]["oa_percent"] for arm in ARMS]
        lines.append(f"| {seed} | " + " | ".join(f"{x:.2f}" for x in values) + " |")
    lines += ["", "## Per-class recall (%)", "",
              "Mean ± sample SD across seeds; classes follow the BiDA loader order.", "",
              "| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        values = np.asarray([runs[(arm, seed)]["target"]["per_class_percent"]
                             for seed in SEEDS])
        lines.append(f"| {arm.upper()} | " + " | ".join(
            mean_sd(values[:, class_id]) for class_id in range(values.shape[1])) + " |")
    lines += ["", "## Representation and collapse audit", "",
              "Cosines are mean off-diagonal values over the four slots, on all target patches.", "",
              "| Arm | Seed | Source val OA (%) | Target slot cosine | Target attention cosine | Predicted classes |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for seed in SEEDS:
            run = runs[(arm, seed)]
            target = run["target"]
            counts = target.get("prediction_counts")
            if counts is None:
                counts = np.asarray(target["confusion_matrix"]).sum(axis=0)
            lines.append(f"| {arm.upper()} | {seed} | "
                         f"{run['source_val']['oa_percent']:.2f} | "
                         f"{offdiag_mean(target['mean_slot_cosine']):.3f} | "
                         f"{offdiag_mean(target['mean_attention_cosine']):.3f} | "
                         f"{sum(np.asarray(counts) > 0)} / {len(counts)} |")
    lines += ["", "## Alignment and consistency diagnostics", "",
              "First/final epoch losses below are raw, unweighted means over source steps.", "",
              "| Arm | First MMD² | Final MMD² | First KL | Final KL |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS[1:]:
        first = [runs[(arm, seed)]["first_epoch_losses"] for seed in SEEDS]
        final = [runs[(arm, seed)]["final_epoch_losses"] for seed in SEEDS]
        lines.append(f"| {arm.upper()} | "
                     f"{np.mean([x['alignment_mmd2'] for x in first]):.6f} | "
                     f"{np.mean([x['alignment_mmd2'] for x in final]):.6f} | "
                     f"{np.mean([x['target_consistency_kl'] for x in first]):.6f} | "
                     f"{np.mean([x['target_consistency_kl'] for x in final]):.6f} |")
    lines += ["", "Inspect `summary.json` in each run for per-class recall, prediction counts,",
              "spatial attention maps, slot cosine, and attention-map cosine.", "",
              "## Conclusion", "",
              "B0 has the highest mean fixed-epoch target OA. B2 does not beat B1 on any seed,",
              "and B4 only modestly improves B2 while remaining well below B0. B3 is below B0",
              "on every seed. B1, B2, and B4 collapse to one predicted class on seed 2101,",
              "with source validation OA near chance. B3 fits source validation but transfers",
              "poorly on two seeds. These observations do not support the proposed slot-wise",
              "alignment and target-consistency gains under the locked V1 settings; they do",
              "not isolate the cause of instability or rule out other implementations.", ""]
    output = ROOT / "experiments" / "slot_uda" / "RESULTS.md"
    output.write_text("\n".join(lines))
    compact = {f"{arm}_{seed}": runs[(arm, seed)] for arm in ARMS for seed in SEEDS}
    (ROOT / "results" / "slot_uda" / "aggregate.json").write_text(
        json.dumps(compact, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
