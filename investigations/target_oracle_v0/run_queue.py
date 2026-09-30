"""Run independent target-oracle V0 cells sequentially on one visible GPU."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", nargs="+", help="Cells such as a0:2100 a2:2102")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=PROJECT / "results" / "v0_target_oracle")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for cell in args.cells:
        arm, seed = cell.split(":", 1)
        if arm not in {"a0", "a1", "a2"} or seed not in {"2100", "2101", "2102"}:
            parser.error(f"Invalid cell: {cell}")
        run = args.out_root / f"{arm}_{seed}"
        log_path = args.out_root / f"{arm}_{seed}.log"
        if run.exists() or log_path.exists():
            raise FileExistsError(f"Existing output for {cell}")
        command = [sys.executable, str(HERE / "run.py"), "--variant", arm, "--seed", seed,
                   "--device", args.device, "--num-workers", str(args.num_workers), "--out", str(run)]
        print(json.dumps({"cell": cell, "run": str(run), "log": str(log_path)}), flush=True)
        with log_path.open("w") as log:
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run / "summary.json").read_text())
        print(json.dumps({"cell": cell, "selected_epoch": result["selected_epoch"],
                          "target_best_oa_percent": result["target_at_best"]["oa_percent"]}), flush=True)


if __name__ == "__main__":
    main()
