"""Launch the unmodified server BiDA main.py in independent NAS workdirs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BIDA = Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_native_entry_v1')
DATA = Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise RuntimeError('Refusing to reuse a nonempty result directory')
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for path in sorted(BIDA.rglob('*.py')):
        relative = path.relative_to(BIDA)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(destination))
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(OUT / 'source_hashes.json', hashes)
    dump(OUT / 'protocol.json', {
        'purpose': 'native main.py reproduction diagnostic, not a new method experiment',
        'entry': str(BIDA / 'main.py'), 'source_unmodified': True,
        'source_provenance': 'server official checkout with previously audited dataset runtime fixes',
        'dataset': str(DATA), 'direction': 'Dioni to Loukia',
        'seeds': [2100, 2101, 2102], 'epochs': 200,
        'selection': 'native target-best every 10 epochs; diagnostic only',
        'fixed200': 'native epoch-200 printed evaluation, without replacing old formal results',
        'extra_source_validation': False, 'python': sys.executable,
        'no_hyperparameter_search': True,
    })
    children = []
    for seed, device in ((2100, 6), (2101, 7), (2102, 6)):
        work = OUT / ('seed_%d' % seed)
        work.mkdir()
        command = [sys.executable, '-u', str(BIDA / 'main.py'),
                   '--model', 'BiDA', '--source_name', 'Dioni', '--target_name', 'Loukia',
                   '--dataset_dir', str(DATA) + '/', '--device', str(device),
                   '--epoch', '200', '--bs', '128', '--lr', '0.01', '--ratio', '0.95',
                   '--patch_size', '13', '--dim', '64', '--depth', '3', '--num_tokens', '4',
                   '--num_workers', '4', '--ema_decay', '0.999', '--lambda1', '0.1',
                   '--lambda2', '1.0', '--log_interval', '10', '--re_ratio', '1',
                   '--loss_type', 'softmax', '--labelsmooth', 'off', '--seed', str(seed)]
        dump(work / 'command.json', command)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        log = (OUT / ('seed_%d.log' % seed)).open('w')
        process = subprocess.Popen(command, cwd=str(work), env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        children.append((process, log, seed, device))
    while True:
        rows = [{'seed': seed, 'device': device, 'pid': process.pid,
                 'exit_code': process.poll()} for process, log, seed, device in children]
        dump(OUT / 'launcher_status.json', {'jobs': rows})
        if all(row['exit_code'] is not None for row in rows):
            break
        time.sleep(10)
    for process, log, seed, device in children:
        log.close()
    if any(row['exit_code'] != 0 for row in rows):
        raise RuntimeError('Native entry failed; inspect logs. No automatic retry.')


if __name__ == '__main__':
    main()
