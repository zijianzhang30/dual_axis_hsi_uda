"""Verify the paired experiment and render its fixed-200 report."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pavia_joint_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def describe(values, digits=2):
    return f'{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}'


def main():
    results, histories = {}, {}
    for arm in ('original', 'joint'):
        for seed in (2100, 2101, 2102):
            folder = OUT / f'{arm}_{seed}'
            result = json.loads((folder / 'results.json').read_text())
            assert sha(folder / 'frozen_manifest.json') == result['frozen_manifest_sha256']
            for item in [*result['checkpoints'], result['prediction']]:
                assert sha(item['path']) == item['sha256']
            history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in history] == list(range(1, 201))
            assert [r['epoch'] for r in result['checkpoints']] == list(range(10, 201, 10))
            results[arm, seed], histories[arm, seed] = result, history
    for seed in (2100, 2101, 2102):
        a, b = results['original', seed], results['joint', seed]
        for key in ('initial_model_sha256', 'sample_counts', 'data_sha256', 'protocol_sha256',
                    'script_sha256', 'model_code_sha256', 'dataset_code_sha256', 'normalization_sha256'):
            assert a['config'][key] == b['config'][key], (seed, key)
        for a, b in zip(histories['original', seed], histories['joint', seed]):
            for key in ('steps', 'batch_sha256', 'rng_sha256'):
                assert a[key] == b[key], (seed, a['epoch'], key)
    metrics = ('oa_percent', 'aa_percent', 'kappa')
    deltas = {key: [results['joint', seed]['target_at_final'][key] -
                   results['original', seed]['target_at_final'][key]
                   for seed in (2100, 2101, 2102)] for key in metrics}
    collapse = {arm: sum(results[arm, seed]['target_at_final']['collapse'] for seed in (2100, 2101, 2102))
                for arm in ('original', 'joint')}
    go = collapse['joint'] <= collapse['original'] and any(
        np.mean(deltas[key]) >= 1 and sum(v > 0 for v in deltas[key]) >= 2
        for key in ('oa_percent', 'aa_percent'))
    summary = {'paired_deltas': deltas, 'collapse_counts': collapse, 'positive_replication_gate': bool(go),
               'runs': {f'{arm}_{seed}': results[arm, seed] for arm, seed in results},
               'paired_init_batch_rng_audit': 'passed all 200 epochs of all 3 seeds'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# PaviaU → PaviaC normalization transfer: fixed epoch 200', '',
             'BiDA-self with original BiDA tokenizer. No SceneShift or additional losses.', '',
             '| Arm | OA (%) | AA (%) | Kappa | Collapse |',
             '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('original', 'joint'):
        values = [results[arm, seed]['target_at_final'] for seed in (2100, 2101, 2102)]
        lines.append(f'| {arm} | ' + ' | '.join(describe([v[key] for v in values], 3 if key == 'kappa' else 2)
                                               for key in metrics) + f' | {collapse[arm]}/3 |')
    lines += ['', 'Mean ± sample std (ddof=1), seeds 2100/2101/2102.', '',
              '## Paired Joint−Original', '', '| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |',
              '| --- | ---: | ---: | ---: |']
    for i, seed in enumerate((2100, 2101, 2102)):
        lines.append(f'| {seed} | {deltas[metrics[0]][i]:+.2f} | {deltas[metrics[1]][i]:+.2f} | {deltas[metrics[2]][i]:+.3f} |')
    lines.append('| Mean ± std | ' + ' | '.join(describe(deltas[k], 3 if k == 'kappa' else 2) for k in metrics) + ' |')
    lines += ['', f'Predeclared positive-replication gate: **{"PASS" if go else "FAIL"}**.',
              'The gate requires ≥2/3 positive paired seeds and ≥1 pp mean improvement in the same OA/AA metric, with no increase in collapse.',
              'Three seeds do not establish statistical significance.', '', '## Individual results', '',
              '| Arm | Seed | OA | AA | Kappa | Class-6 share (%) | Max class share (%) | Zero-recall classes | Source-val OA |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for (arm, seed), r in results.items():
        m = r['target_at_final']
        zero = [i + 1 for i, v in enumerate(m['per_class_percent']) if v == 0]
        lines.append(f"| {arm} | {seed} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} | {m['kappa']:.3f} | {100*m['prediction_shares'][5]:.2f} | {100*max(m['prediction_shares']):.2f} | {zero or 'none'} | {r['source_val_at_final']['oa_percent']:.2f} |")
    for arm in ('original', 'joint'):
        lines += ['', f'## {arm}: per-class recall (%)', '', '| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed in (2100, 2101, 2102):
            lines.append(f'| {seed} | ' + ' | '.join(f'{v:.2f}' for v in results[arm, seed]['target_at_final']['per_class_percent']) + ' |')
        lines += ['', 'Prediction distribution (%), classes 1–7:', '', '| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed in (2100, 2101, 2102):
            lines.append(f'| {seed} | ' + ' | '.join(f'{100*v:.2f}' for v in results[arm, seed]['target_at_final']['prediction_shares']) + ' |')
    lines += ['', '## Cost and protocol audit', '', '| Arm | Parameters | Training seconds | Source-val OA |',
              '| --- | ---: | ---: | ---: |']
    for arm in ('original', 'joint'):
        rows = [results[arm, seed] for seed in (2100, 2101, 2102)]
        lines.append(f"| {arm} | {rows[0]['config']['trainable_parameters']} | {describe([r['train_seconds'] for r in rows])} | {describe([r['source_val_at_final']['oa_percent'] for r in rows])} |")
    ratio = [results['joint', s]['train_seconds'] / results['original', s]['train_seconds'] for s in (2100, 2101, 2102)]
    lines += ['', f'Paired wall-time ratio Joint/Original: {describe(ratio, 3)}; concurrent GPU jobs, not an isolated speed benchmark.',
              f"Dataset sample counts: `{results['original', 2100]['config']['sample_counts']}`.",
              f"Updates per epoch: {histories['original', 2100][0]['steps']}.",
              'Initial tensors, all training batches and pre-forward CPU/CUDA RNG hashes match within each paired seed for all 200 epochs.',
              'All 120 checkpoint hashes and six frozen target-prediction hashes verified.',
              'This preserves the Houston 95% source split, normband, and target-mask sampler; it is not an official Pavia protocol reproduction.',
              'Only the 102-band input dimension is adapted. No target-based checkpoint selection or tuning.',
              'Full artifacts and machine-readable metrics: `results/pavia_joint_v1/summary.json`.']
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
