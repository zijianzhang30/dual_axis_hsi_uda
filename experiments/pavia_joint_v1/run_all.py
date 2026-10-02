"""Durable six-job runner; save process status and results after completion."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pavia_joint_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    handles, jobs = [], []
    for gpu, seed in zip((5, 6, 7), (2100, 2101, 2102)):
        for arm in ('original', 'joint'):
            name = f'{arm}_{seed}'
            log = (OUT / f'{name}.log').open('x')
            handles.append(log)
            command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')),
                       '--arm', arm, '--seed', str(seed), '--device', str(gpu), '--out', str(OUT / name)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((name, process))
    status = {name: {'pid': process.pid, 'state': 'running'} for name, process in jobs}
    (OUT / 'runner_status.json').write_text(json.dumps(status, indent=2) + '\n')
    for name, process in jobs:
        status[name].update(state='finished', exit_code=process.wait())
        (OUT / 'runner_status.json').write_text(json.dumps(status, indent=2) + '\n')
    for log in handles:
        log.close()
    if any(v['exit_code'] != 0 for v in status.values()):
        raise RuntimeError('A training run failed; inspect individual logs')
    report = subprocess.run([sys.executable, str(Path(__file__).with_name('summarize.py'))],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    Path(__file__).with_name('RESULTS.md').write_text(report.stdout)
    print(report.stdout, flush=True)


if __name__ == '__main__':
    main()
