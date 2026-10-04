"""Prepare an immutable native snapshot and wait for current jobs; no retries."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/zhangzj26/dual_axis_hsi_uda')
SOURCE = Path('/home/zhangzj26/TGRS_MLUDA-2024-main')
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/mluda_pavia_native_v1')
PRIOR = OUT.parent / 'pavia_zscore_v1'
DATA = Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def alive(pid):
    path = Path('/proc') / str(pid) / 'cmdline'
    return path.exists() and b'pavia_zscore_v1/train.py' in path.read_bytes()


def main():
    OUT.mkdir(exist_ok=False)
    snapshot = OUT / 'native_snapshot'
    snapshot.mkdir()
    hashes = {}
    for path in list(SOURCE.glob('*.py')) + [SOURCE / 'README.md']:
        shutil.copy2(str(path), str(snapshot / path.name))
        hashes[path.name] = sha(path)
        assert sha(snapshot / path.name) == hashes[path.name]
    (snapshot / 'datasets').mkdir()
    (snapshot / 'datasets/Pavia').symlink_to(DATA, target_is_directory=True)
    data_hashes = {name: sha(DATA / name) for name in
                  ('paviaU.mat', 'paviaU_gt_7.mat', 'pavia.mat', 'pavia_gt_7.mat')}
    for path in Path(__file__).parent.glob('*'):
        if path.is_file():
            shutil.copy2(str(path), str(OUT / path.name))
    dump(OUT / 'manifest.json', {'official_directory': str(SOURCE),
         'source_sha256': hashes, 'data_directory': str(DATA), 'data_sha256': data_hashes,
         'source_patch': 'none', 'native_config_changes': 'none',
         'gpu': 6, 'wait_for_pids': [619134, 619135], 'free_bytes': shutil.disk_usage(OUT).free})
    link = ROOT / 'results/mluda_pavia_native_v1'
    if link.exists() or link.is_symlink():
        raise RuntimeError('Existing project result link: do not overwrite')
    link.symlink_to(OUT, target_is_directory=True)
    while any(alive(pid) for pid in (619134, 619135)):
        dump(OUT / 'status.json', {'state': 'waiting_for_current_pavia_training',
                                 'checked_at': time.time()})
        time.sleep(20)
    for arm in ('B', 'C'):
        folder = PRIOR / (arm + '_2100')
        history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
        assert history[-1]['epoch'] == 200, 'Prior training did not finish; do not launch'
        assert (folder / 'frozen_manifest.json').exists(), 'Prior predictions not frozen'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='6', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
    with (OUT / 'native.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(OUT / 'run.py')],
                                   cwd=str(snapshot), env=env, stdout=log, stderr=subprocess.STDOUT)
        dump(OUT / 'status.json', {'state': 'running', 'pid': process.pid,
                                 'physical_gpu': 6, 'started_at': time.time()})
        code = process.wait()
    dump(OUT / 'status.json', {'state': 'completed' if code == 0 else 'failed',
                             'pid': process.pid, 'exit_code': code, 'ended_at': time.time(),
                             'automatic_retries': False})


if __name__ == '__main__':
    main()
