"""Aggregate the paired A1 and A1+diversity runs after all six finish."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2100, 2101, 2102)
ARMS = ("d0", "d1")


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def offdiag(matrix):
    matrix = np.asarray(matrix)
    return float(matrix[np.triu_indices(len(matrix), 1)].mean())


def border_mass(attention):
    attention = np.asarray(attention)
    side = int(np.sqrt(attention.shape[1]))
    if side * side != attention.shape[1]:
        raise ValueError("Attention map is not square")
    maps = attention.reshape(len(attention), side, side)
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return maps[:, mask].sum(axis=1)


def load_runs():
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = ROOT / "results" / "slot_diversity" / f"{arm}_{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
            if len(history) != 200 or summary["config"]["checkpoint_selection"] != "fixed_epoch_200":
                raise ValueError(f"Incomplete or nonformal run: {folder}")
            runs[(arm, seed)] = {"summary": summary, "history": history}
    return runs


def main():
    runs = load_runs()
    lines = ["# A1 spatial attention diversity gate", "",
             "Houston13 → Houston18; fixed epoch 200, three optimization seeds.",
             "D0 is fresh A1; D1 adds source attention diversity with fixed weight 0.1.",
             "The target GT mask defines eligible target training centers under the Strict BiDA",
             "protocol; target label values enter neither loss nor checkpoint selection.", "",
             "| Arm | Target OA (%) | Target AA (%) | Kappa |",
             "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        targets = [runs[(arm, seed)]["summary"]["target_at_final"] for seed in SEEDS]
        lines.append(f"| {arm.upper()} | {mean_sd([x['oa_percent'] for x in targets])} | "
                     f"{mean_sd([x['aa_percent'] for x in targets])} | "
                     f"{mean_sd([x['kappa'] for x in targets])} |")
    differences = [runs[("d1", seed)]["summary"]["target_at_final"]["oa_percent"] -
                   runs[("d0", seed)]["summary"]["target_at_final"]["oa_percent"]
                   for seed in SEEDS]
    lines += ["", f"Paired target OA difference D1 − D0: **{mean_sd(differences)} points**.", "",
              "## Per-seed outcome and slot separation", "",
              "Cosines are mean off-diagonal values across the four slots.", "",
              "| Arm | Seed | Source val OA | Target OA | Target AA | Source attention cosine | Target attention cosine | Source token cosine | Target token cosine |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for seed in SEEDS:
            summary = runs[(arm, seed)]["summary"]
            source = summary["source_val_at_final"]
            target = summary["target_at_final"]
            lines.append(f"| {arm.upper()} | {seed} | {source['oa_percent']:.2f} | "
                         f"{target['oa_percent']:.2f} | {target['aa_percent']:.2f} | "
                         f"{offdiag(source['mean_attention_cosine']):.3f} | "
                         f"{offdiag(target['mean_attention_cosine']):.3f} | "
                         f"{offdiag(source['mean_slot_cosine']):.3f} | "
                         f"{offdiag(target['mean_slot_cosine']):.3f} |")
    lines += ["", "## Attention location audit", "",
              "The border is the outermost row or column of the 13×13 patch; it covers",
              "48/169 = 28.4% of spatial positions. Values are fractions of target",
              "attention mass in that border, by query slot.", "",
              "| Arm | Seed | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Mean |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for seed in SEEDS:
            target = runs[(arm, seed)]["summary"]["target_at_final"]
            mass = border_mass(target["mean_spatial_attn"])
            lines.append(f"| {arm.upper()} | {seed} | " + " | ".join(
                f"{value:.3f}" for value in mass) + f" | {mass.mean():.3f} |")
    lines += ["", "## Per-class target recall (%)", "",
              "Mean ± sample SD across seeds; classes follow the BiDA loader order.", "",
              "| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        values = np.asarray([runs[(arm, seed)]["summary"]["target_at_final"]["per_class_percent"]
                             for seed in SEEDS])
        lines.append(f"| {arm.upper()} | " + " | ".join(
            mean_sd(values[:, class_id]) for class_id in range(values.shape[1])) + " |")
    lines += ["", "## Training loss diagnostics", "",
              "Diversity is the raw mean squared off-diagonal attention cosine.", "",
              "| Arm | First diversity | Final diversity | Final source CE |",
              "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        first = [runs[(arm, seed)]["history"][0]["losses"] for seed in SEEDS]
        final = [runs[(arm, seed)]["history"][-1]["losses"] for seed in SEEDS]
        lines.append(f"| {arm.upper()} | "
                     f"{np.mean([x['diversity'] for x in first]):.4f} | "
                     f"{np.mean([x['diversity'] for x in final]):.4f} | "
                     f"{np.mean([x['source_ce'] for x in final]):.4f} |")
    lines += ["", "Per-run summaries contain confusion matrices, per-class recall, prediction",
              "counts, average spatial attention maps, and pairwise slot similarity matrices.",
              "Attention-map figures for all seeds are in `results/slot_diversity/attention_maps_<seed>.png`.",
              "The target class-example figure for seed 2100 uses GT only for post-hoc",
              "sample selection: `results/slot_diversity/class_examples_d1_2100.png`.", "",
              "## Interpretation", "",
              "D1 sharply lowers attention-map and token cosine on source and target while",
              "retaining perfect source validation OA at epoch 200. Mean target OA is 1.10",
              "points below the fresh D0 control, with a 3.39-point drop in seed 2102.",
              "Many D1 queries focus on fixed border rows or columns, as the maps and",
              "border-mass audit show. This establishes spatial separation but does not",
              "establish different semantic roles. The proposed specialization gate is",
              "therefore only partially met; proceeding directly to same-slot cross-domain",
              "attention would assume semantics that this experiment has not verified.", ""]
    output = ROOT / "experiments" / "slot_diversity" / "RESULTS.md"
    output.write_text("\n".join(lines))
    (ROOT / "results" / "slot_diversity" / "aggregate.json").write_text(
        json.dumps({f"{arm}_{seed}": runs[(arm, seed)]["summary"]
                    for arm in ARMS for seed in SEEDS}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
