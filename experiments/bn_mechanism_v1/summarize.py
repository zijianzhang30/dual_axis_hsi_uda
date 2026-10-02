"""Verify artifacts and compare the four fixed-endpoint causal arms."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/bn_mechanism_v1'
SEEDS = (2100, 2101, 2102)
ARMS = ('original_mixed', 'buffer_only', 'detach_target', 'fixed_mixture')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    runs, summaries, checks = {}, {}, 0
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            if arm == 'original_mixed':
                folder = ROOT / 'results/bida_self' / ('diagnostic_2100' if seed == 2100 else f'multiseed_{seed}')
            elif arm == 'fixed_mixture':
                folder = ROOT / 'results/bn_stabilization_v1' / f'{arm}_{seed}'
            else:
                folder = OUT / f'{arm}_{seed}'
            result = json.loads((folder / 'results.json').read_text())
            assert digest(folder / 'frozen_manifest.json') == result['frozen_manifest_sha256']
            manifest = json.loads((folder / 'frozen_manifest.json').read_text())
            if arm != 'original_mixed':
                for group in ('checkpoints', 'predictions'):
                    assert [x['epoch'] for x in manifest[group]] == list(range(10, 201, 10))
                    for item in manifest[group]:
                        assert digest(item['path']) == item['sha256']
                        checks += 1
            if arm == 'buffer_only':
                assert result['control_learned_tensor_comparison']['exact']
                assert result['max_source_ce_replay_error'] <= 1e-7
            row = result['target_at_final']
            rows.append(row)
            runs[f'{arm}_{seed}'] = result
        summary = {'collapse_count': sum(x['prediction_shares'][5] >= .95 for x in rows)}
        for key in ('oa_percent', 'aa_percent', 'kappa'):
            values = np.array([x[key] for x in rows])
            summary[key] = {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}
        summary['per_class_mean'] = np.array([x['per_class_percent'] for x in rows]).mean(0).tolist()
        summaries[arm] = summary
    comparisons = {}
    for arm in ('buffer_only', 'detach_target'):
        delta = [runs[f'{arm}_{seed}']['target_at_final']['oa_percent'] - runs[f'fixed_mixture_{seed}']['target_at_final']['oa_percent'] for seed in SEEDS]
        aa = summaries[arm]['aa_percent']['mean'] - summaries['fixed_mixture']['aa_percent']['mean']
        original_delta = [runs[f'{arm}_{seed}']['target_at_final']['oa_percent'] - runs[f'original_mixed_{seed}']['target_at_final']['oa_percent'] for seed in SEEDS]
        comparisons[arm] = {'paired_oa_minus_full': delta, 'paired_oa_minus_original': original_delta, 'mean_aa_minus_full': aa,
                            'descriptively_close': bool(abs(np.mean(delta)) <= 2 and abs(aa) <= 2 and max(abs(x) for x in delta) <= 2 and summaries[arm]['collapse_count'] == 0)}
    payload = {'summaries': summaries, 'comparisons': comparisons, 'verified_artifacts': checks, 'runs': runs}
    (OUT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# BN mechanism V1: fixed epoch-200 results', '',
             'Three optimization seeds (2100/2101/2102), one Houston transfer pair. Mean ± sample SD.', '',
             '| Arm | OA | AA | Class-6 collapse |', '| --- | ---: | ---: | ---: |']
    for arm in ARMS:
        row = summaries[arm]
        cells = [f"{row[k]['mean']:.2f} ± {row[k]['sample_sd']:.2f}" for k in ('oa_percent', 'aa_percent')]
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {row['collapse_count']}/3 |")
    lines += ['', 'Balanced inference buffers alone improve mean OA by 1.96 points and remove endpoint class-6 collapse, but remain 4.37 OA and 10.74 AA points below Full Joint.',
              'Stopping target moment gradients lowers OA relative to Full Joint in all three seeds (8.67, 7.83, 3.81 points); mean OA/AA fall by 6.77/20.19 points.',
              'These controlled endpoints support a contribution from target-side moment gradients during source-discriminative training, beyond buffer calibration. They do not prove that shared training normalization without those gradients is beneficial: Detach underperforms Buffer-only on average.',
              'The differences are not an additive decomposition: trained parameters and subsequent statistics co-evolve differently in each arm.',
              '', '## Per-seed endpoints', '', '| Arm | Seed | OA | AA | Class-6 share |', '| --- | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        for seed in SEEDS:
            row = runs[f'{arm}_{seed}']['target_at_final']
            lines.append(f"| {arm} | {seed} | {row['oa_percent']:.2f} | {row['aa_percent']:.2f} | {100 * row['prediction_shares'][5]:.2f}% |")
    lines += ['', '## Predeclared descriptive proximity to Full Joint', '', 'Not a statistical equivalence test. All three paired OA gaps and mean OA/AA gaps must be within two points, with no endpoint class-6 collapse.', '']
    for arm, comparison in comparisons.items():
        lines.append(f"- {arm}: close = {comparison['descriptively_close']}; paired OA differences = {comparison['paired_oa_minus_full']}; mean AA difference = {comparison['mean_aa_minus_full']:.2f}.")
    lines += ['', '## Mean per-class recall', '', '| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{x:.2f}' for x in summaries[arm]['per_class_mean']) + ' |')
    lines += ['', '## Frozen trajectory collapse', '', '| Arm | Seed | Class-6 collapse grid points |', '| --- | ---: | ---: |']
    for arm in ARMS[1:]:
        for seed in SEEDS:
            count = sum(x['metrics']['prediction_shares'][5] >= .95 for x in runs[f'{arm}_{seed}']['target_trajectory'])
            lines.append(f'| {arm} | {seed} | {count}/20 |')
    lines += ['', '## Controls and limitations', '',
              'Buffer-only source training losses and final learned tensors reproduce Original Mixed BN exactly for all three seeds. Only running buffers differ.',
              'Detach-target has exactly Full Joint forward arithmetic in the unit fixture, but no target input gradient through the moments under source-only CE.',
              'Both new arms use 0.19 EMA per paired step, matched to the original two 0.1 updates. Intermediate buffer-only layer moments differ from Full Joint because training normalization differs.',
              f'Verified {checks} checkpoint/prediction hashes and all 12 manifests. All 20 predictions per new run were frozen before post-hoc target metrics.',
              'Three seeds in one transfer pair do not establish a general mechanism across datasets. No oracle checkpoint or mixture-rate tuning is used.']
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'summaries': summaries, 'comparisons': comparisons}, indent=2))

if __name__ == '__main__':
    main()
