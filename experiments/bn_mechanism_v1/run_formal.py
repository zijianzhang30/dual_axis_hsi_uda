"""Run one predeclared BN arm sequentially for the three fixed seeds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("buffer_only", "detach_target"), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out_root = root / "results/bn_mechanism_v1"
    out_root.mkdir(parents=True, exist_ok=True)
    for seed in (2100, 2101, 2102):
        folder = out_root / f"{args.arm}_{seed}"
        if (folder / "results.json").exists():
            print(json.dumps({"complete_existing": str(folder)}), flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Incomplete run requires inspection: {folder}")
        command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
                   "--arm", args.arm, "--seed", str(seed), "--device", args.device, "--out", str(folder)]
        print(json.dumps({"starting": args.arm, "seed": seed, "device": args.device}), flush=True)
        with (out_root / f"{args.arm}_{seed}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f"Run failed: {folder}")


if __name__ == "__main__":
    main()
