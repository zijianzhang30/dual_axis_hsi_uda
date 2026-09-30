"""Run paired C0/C1 set-level coupled attention study."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
ARMS = ("c0", "c1")
SEEDS = (2100, 2101, 2102)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "set_coupled")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for arm in args.arm:
        for seed in args.seed:
            folder = args.out_root / f"{arm}_{seed}"
            log_path = args.out_root / f"{arm}_{seed}.log"
            if folder.exists() or log_path.exists():
                raise FileExistsError(f"Existing output needs review: {folder} or {log_path}")
            command = [sys.executable, str(STUDY / "train.py"), "--arm", arm,
                       "--seed", str(seed), "--epochs", "200", "--device", args.device,
                       "--num-workers", str(args.num_workers), "--out", str(folder)]
            print(json.dumps({"arm": arm, "seed": seed, "output": str(folder)}), flush=True)
            with log_path.open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            summary = json.loads((folder / "summary.json").read_text())
            print(json.dumps({"arm": arm, "seed": seed,
                              "target_oa": summary["target_at_final"]["oa_percent"]}), flush=True)


if __name__ == "__main__":
    main()
