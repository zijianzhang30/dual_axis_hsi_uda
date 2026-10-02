"""Independent epoch-100 diagnostic; never alters running training processes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pavia_joint_v1'
DIAGNOSTIC = OUT / 'diagnostic_epoch_100'
SEEDS = (2100, 2101, 2102)


def ready():
    for arm in ('original', 'joint'):
        for seed in SEEDS:
            folder = OUT / f'{arm}_{seed}'
            history = folder / 'history.jsonl'
            lines = history.read_text().splitlines() if history.exists() else []
            try:
                epoch = json.loads(lines[-1])['epoch'] if lines else 0
            except json.JSONDecodeError:
                return False
            if epoch < 100:
                log = (OUT / f'{arm}_{seed}.log').read_text()
                if 'Traceback' in log:
                    raise RuntimeError(f'Training failure: {arm}_{seed}')
                return False
            if not (folder / 'checkpoints/epoch_100.pth').exists():
                raise RuntimeError(f'Missing epoch-100 checkpoint: {folder}')
    return True


def fmt(values, digits=2):
    return f'{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=5)
    args = parser.parse_args()
    if DIAGNOSTIC.exists():
        raise RuntimeError('Diagnostic output already exists; refusing overwrite')
    print('Waiting for all six completed epoch-100 checkpoints.', flush=True)
    while not ready():
        time.sleep(10)
    DIAGNOSTIC.mkdir()
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.set_device(device)
    loader_args = argparse.Namespace(seed=2100, num_workers=4,
                    dataset_dir=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia'))
    train.seed_worker(2100)
    _, _, _, loader = train.make_loaders(loader_args)
    records = []
    for arm in ('original', 'joint'):
        for seed in SEEDS:
            checkpoint = OUT / f'{arm}_{seed}/checkpoints/epoch_100.pth'
            checkpoint_hash = train.helper.file_hash(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert state['epoch'] == 100
            assert state['config']['sample_counts']['target_eval'] == len(loader.dataset)
            model = train.BiDAnet(n_bands=102, num_classes=7, num_tokens=4, dim=64, depth=3)
            if arm == 'joint':
                train.install(model, 'fixed_mixture')
            model.load_state_dict(state['model'], strict=True)
            del state
            model.to(device)
            logits, predictions = train.helper.predict(model, loader, device)
            path = DIAGNOSTIC / f'{arm}_{seed}.npz'
            np.savez_compressed(path, logits=logits, predictions=predictions)
            assert train.helper.file_hash(checkpoint) == checkpoint_hash
            records.append({'arm': arm, 'seed': seed, 'checkpoint_path': str(checkpoint),
                            'checkpoint_sha256': checkpoint_hash, 'prediction_path': str(path),
                            'prediction_sha256': train.helper.file_hash(path)})
            print(json.dumps({'frozen_prediction': f'{arm}_{seed}', 'epoch': 100}), flush=True)
            del model, logits, predictions
    manifest_path = DIAGNOSTIC / 'frozen_manifest.json'
    manifest_path.write_text(json.dumps({'epoch': 100, 'records': records,
        'addendum_sha256': train.helper.file_hash(Path(__file__).with_name('EPOCH100_ADDENDUM.md')),
        'evaluation_script_sha256': train.helper.file_hash(Path(__file__))}, indent=2) + '\n')
    manifest_hash = train.helper.file_hash(manifest_path)
    # All six predictions are frozen before target values are collected for scoring.
    labels = np.concatenate([y.numpy() for _, y in loader])
    results = {}
    for record in records:
        with np.load(record['prediction_path']) as artifact:
            metrics = train.helper.metrics(labels, artifact['predictions'], artifact['logits'], 7)
        metrics['collapse'] = max(metrics['prediction_shares']) >= .95
        results[f"{record['arm']}_{record['seed']}"] = metrics
    keys = ('oa_percent', 'aa_percent', 'kappa')
    deltas = {k: [results[f'joint_{s}'][k]-results[f'original_{s}'][k] for s in SEEDS] for k in keys}
    (DIAGNOSTIC / 'results.json').write_text(json.dumps({'epoch': 100,
        'endpoint': 'user-requested mid-training diagnostic, not model selection',
        'frozen_manifest_sha256': manifest_hash, 'runs': results,
        'paired_joint_minus_original': deltas}, indent=2) + '\n')
    lines = ['# Pavia epoch-100 diagnostic', '',
        'User-requested mid-training diagnostic, not the primary result. Fixed epoch 200 remains primary; no target-based selection or tuning.', '',
        '| Arm | OA (%) | AA (%) | Kappa | Collapse |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('original', 'joint'):
        rows = [results[f'{arm}_{s}'] for s in SEEDS]
        lines.append(f'| {arm} | ' + ' | '.join(fmt([r[k] for r in rows], 3 if k=='kappa' else 2) for k in keys)
                     + f" | {sum(r['collapse'] for r in rows)}/3 |")
    lines += ['', 'Mean ± sample std, ddof=1.', '',
        '| Arm | Seed | OA | AA | Kappa | Class-6 share (%) | Max class share (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('original', 'joint'):
        for seed in SEEDS:
            m = results[f'{arm}_{seed}']
            lines.append(f"| {arm} | {seed} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} | {m['kappa']:.3f} | {100*m['prediction_shares'][5]:.2f} | {100*max(m['prediction_shares']):.2f} |")
    lines += ['', '## Paired Joint−Original', '', '| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |',
              '| --- | ---: | ---: | ---: |']
    for i, s in enumerate(SEEDS):
        lines.append(f'| {s} | ' + ' | '.join(f'{deltas[k][i]:+.3f}' for k in keys) + ' |')
    lines.append('| Mean ± std | ' + ' | '.join(fmt(deltas[k], 3 if k=='kappa' else 2) for k in keys) + ' |')
    for name, m in results.items():
        lines += ['', f'## {name}: classes 1–7', '',
                  '| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
                  '| Recall | ' + ' | '.join(f'{v:.2f}' for v in m['per_class_percent']) + ' |',
                  '| Prediction share | ' + ' | '.join(f'{100*v:.2f}' for v in m['prediction_shares']) + ' |']
    lines += ['', f'Frozen manifest SHA256: `{manifest_hash}`.',
              'Training processes, checkpoints and the original frozen protocol were not modified.']
    report = '\n'.join(lines) + '\n'
    Path(__file__).with_name('DIAGNOSTIC_EPOCH100.md').write_text(report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
