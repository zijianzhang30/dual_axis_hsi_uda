"""Aggregate final-epoch metrics across paired seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories or summary.json files")
    args = parser.parse_args()
    groups = {}
    for path in args.runs:
        # Shell globs such as results/v0/a0_* also include per-run .log files.
        if not path.is_dir() and path.name != "summary.json":
            continue
        path = path / "summary.json" if path.is_dir() else path
        result = json.loads(path.read_text())
        if "target_at_final" not in result:
            raise ValueError(f"No target final evaluation in {path}")
        variant = result["config"]["variant"]
        seed = result["config"]["seed"]
        if seed in groups.setdefault(variant, {}):
            raise ValueError(f"Duplicate {variant} seed {seed}")
        groups[variant][seed] = result["target_at_final"]
    if not groups:
        parser.error("No run summaries found")
    report = {}
    for variant, seeds in sorted(groups.items()):
        runs = list(seeds.values())
        entry = {"seeds": sorted(seeds)}
        for key in ("oa_percent", "aa_percent", "kappa"):
            values = [run[key] for run in runs]
            entry[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        per_class = np.asarray([run["per_class_percent"] for run in runs])
        entry["per_class_percent"] = {"mean": per_class.mean(0).tolist(),
                                      "std": per_class.std(0, ddof=1).tolist() if len(runs) > 1 else np.zeros(per_class.shape[1]).tolist()}
        report[variant] = entry
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
