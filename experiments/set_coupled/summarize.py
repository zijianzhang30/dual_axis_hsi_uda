"""Aggregate paired C0/C1 fixed-epoch results."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("c0", "c1")
SEEDS = (2100, 2101, 2102)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def main():
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = ROOT / "results" / "set_coupled" / f"{arm}_{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
            if len(history) != 200 or summary["config"]["checkpoint_selection"] != "fixed_epoch_200":
                raise ValueError(f"Incomplete run: {folder}")
            runs[(arm, seed)] = {"summary": summary, "history": history}
    lines = ["# Set-level coupled cross-domain retrieval", "",
             "Houston13 → Houston18, fixed epoch 200, three optimization seeds.",
             "C0 is the paired A1 self control. C1 adds shared bidirectional set-level",
             "cross-attention, averaged source self/cross CE, and detached target",
             "cross-to-self KL. Inference uses the A1 target self branch only.", "",
             "| Arm | Target OA (%) | Target AA (%) | Kappa |",
             "| --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        targets = [runs[(arm, seed)]["summary"]["target_at_final"] for seed in SEEDS]
        lines.append(f"| {arm.upper()} | {mean_sd([x['oa_percent'] for x in targets])} | "
                     f"{mean_sd([x['aa_percent'] for x in targets])} | "
                     f"{mean_sd([x['kappa'] for x in targets])} |")
    differences = [runs[("c1", seed)]["summary"]["target_at_final"]["oa_percent"] -
                   runs[("c0", seed)]["summary"]["target_at_final"]["oa_percent"]
                   for seed in SEEDS]
    lines += ["", f"Paired target OA difference C1 − C0: **{mean_sd(differences)} points**.", "",
              "## Per-seed outcomes and mechanism diagnostics", "",
              "Entropy is measured over `batch × 4 = 512` opposite-domain tokens during",
              "full batches; the uniform maximum is ln(512) = 6.238.", "",
              "| Seed | C0 OA | C1 OA | Difference | C0 AA | C1 AA | C1 source val OA | Final KL | Final disagreement (%) | Target→source entropy | Source→target entropy |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in SEEDS:
        c0 = runs[("c0", seed)]["summary"]["target_at_final"]
        c1_run = runs[("c1", seed)]
        c1 = c1_run["summary"]["target_at_final"]
        final = c1_run["history"][-1]
        lines.append(f"| {seed} | {c0['oa_percent']:.2f} | {c1['oa_percent']:.2f} | "
                     f"{c1['oa_percent'] - c0['oa_percent']:+.2f} | "
                     f"{c0['aa_percent']:.2f} | {c1['aa_percent']:.2f} | "
                     f"{c1_run['summary']['source_val_at_final']['oa_percent']:.2f} | "
                     f"{final['losses']['target_distill_kl']:.6f} | "
                     f"{final['target_self_cross_disagreement_percent']:.3f} | "
                     f"{final['losses']['target_to_source_entropy']:.3f} | "
                     f"{final['losses']['source_to_target_entropy']:.3f} |")
    lines += ["", "## Per-class target recall (%)", "",
              "Mean ± sample SD across seeds; classes follow the BiDA loader order.", "",
              "| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        values = np.asarray([runs[(arm, seed)]["summary"]["target_at_final"]["per_class_percent"]
                             for seed in SEEDS])
        lines.append(f"| {arm.upper()} | " + " | ".join(
            mean_sd(values[:, class_id]) for class_id in range(values.shape[1])) + " |")
    lines += ["", "Raw histories retain source self/cross CE, target KL, disagreement,",
              "attention entropy, and alpha at every epoch.", ""]
    output = ROOT / "experiments" / "set_coupled" / "RESULTS.md"
    output.write_text("\n".join(lines))
    (ROOT / "results" / "set_coupled" / "aggregate.json").write_text(
        json.dumps({f"{arm}_{seed}": runs[(arm, seed)]
                    for arm in ARMS for seed in SEEDS}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
