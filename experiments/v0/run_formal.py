"""Run the frozen V0 ablation matrix without overwriting existing runs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
ARMS = ("a0", "a1", "a2", "a3")
SEEDS = (2100, 2101, 2102)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--bida-root", type=Path)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "v0")
    parser.add_argument("--skip-target-eval", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for arm in args.variant:
        for seed in args.seed:
            output = args.out_root / f"{arm}_{seed}"
            if output.exists():
                raise FileExistsError(f"Existing run requires explicit review: {output}")
            command = [sys.executable, str(STUDY / "train.py"), "--variant", arm,
                       "--seed", str(seed), "--epochs", "200", "--device", args.device,
                       "--num-workers", str(args.num_workers), "--out", str(output)]
            if args.dataset_dir:
                command.extend(("--dataset-dir", str(args.dataset_dir)))
            if args.bida_root:
                command.extend(("--bida-root", str(args.bida_root)))
            if args.skip_target_eval:
                command.append("--skip-target-eval")
            log_path = args.out_root / f"{arm}_{seed}.log"
            if log_path.exists():
                raise FileExistsError(f"Existing log requires explicit review: {log_path}")
            print(json.dumps({"variant": arm, "seed": seed, "output": str(output),
                              "log": str(log_path)}), flush=True)
            with log_path.open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / "summary.json").read_text())
            target = summary.get("target_at_final", {})
            print(json.dumps({"variant": arm, "seed": seed,
                              "target_self_oa_percent": target.get("oa_percent")}), flush=True)


if __name__ == "__main__":
    main()
