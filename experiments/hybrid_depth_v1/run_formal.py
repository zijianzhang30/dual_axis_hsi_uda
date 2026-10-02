"""Run assigned seeds sequentially without changing experiment settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(2100, 2101, 2102), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    outputs = root / 'results/hybrid_depth_v1'
    outputs.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        folder = outputs / f'formal_{seed}'
        if (folder / 'results.json').exists():
            print(json.dumps({'complete_existing': str(folder)}), flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Incomplete run requires inspection: {folder}')
        command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')), '--seed', str(seed), '--device', args.device, '--out', str(folder)]
        print(json.dumps({'starting': seed, 'device': args.device}), flush=True)
        with (outputs / f'formal_{seed}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'Run failed: {folder}')

if __name__ == '__main__':
    main()
