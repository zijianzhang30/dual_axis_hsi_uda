"""Verify frozen artifacts; separate fixed endpoints from target-oracle diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hybrid_v2'
SEEDS = (2100, 2101, 2102)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def aggregate(rows):
    result = {}
    for key in ('oa_percent', 'aa_percent', 'kappa'):
        values = np.array([row[key] for row in rows])
        result[key] = {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}
    recalls = np.array([row['per_class_percent'] for row in rows])
    result['recall_mean'] = recalls.mean(0).tolist()
    result['recall_sample_sd'] = recalls.std(0, ddof=1).tolist()
    result['class6_collapse_count'] = sum(row['prediction_shares'][5] >= .95 for row in rows)
    return result

def main():
    runs, controls, checked = {}, {}, 0
    for seed in SEEDS:
        folder = OUT / f'formal_{seed}'
        result = json.loads((folder / 'results.json').read_text())
        manifest_path = folder / 'frozen_manifest.json'
        assert digest(manifest_path) == result['frozen_manifest_sha256']
        manifest = json.loads(manifest_path.read_text())
        config = json.loads((folder / 'config.json').read_text())
        assert config == manifest['config']
        assert digest(Path(__file__).with_name('PROTOCOL.md')) == config['protocol_sha256']
        assert digest(Path(__file__).with_name('train.py')) == config['train_script_sha256']
        assert digest(ROOT / 'experiments/bn_stabilization_v1/normalization.py') == config['joint_script_sha256']
        for group in ('checkpoints', 'predictions'):
            assert [row['epoch'] for row in manifest[group]] == list(range(10, 201, 10))
            for row in manifest[group]:
                assert digest(row['path']) == row['sha256']
                checked += 1
        history = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
        assert len(history) == 200
        best = max(result['target_trajectory'], key=lambda x: x['metrics']['oa_percent'])
        assert best['epoch'] == result['target_oracle']['best_epoch']
        assert best['metrics'] == result['target_oracle']['metrics']
        result['source_val_final'] = history[-1]['source_val']
        result['max_class_share_final'] = max(result['target_at_final']['prediction_shares'])
        result['recall_coverage_final'] = sum(x > 0 for x in result['target_at_final']['per_class_percent'])
        runs[str(seed)] = result
        control_folder = ROOT / 'results/bn_stabilization_v1' / f'fixed_mixture_{seed}'
        control = json.loads((control_folder / 'results.json').read_text())
        assert digest(control_folder / 'frozen_manifest.json') == control['frozen_manifest_sha256']
        controls[str(seed)] = control
    fixed = aggregate([runs[str(seed)]['target_at_final'] for seed in SEEDS])
    baseline = aggregate([controls[str(seed)]['target_at_final'] for seed in SEEDS])
    oracle = aggregate([runs[str(seed)]['target_oracle']['metrics'] for seed in SEEDS])
    paired = {str(seed): runs[str(seed)]['hybrid_minus_full_joint_final'] for seed in SEEDS}
    gate = {'mean_fixed_oa_higher_than_full_joint': fixed['oa_percent']['mean'] > baseline['oa_percent']['mean'],
            'no_fixed_class6_collapse': fixed['class6_collapse_count'] == 0}
    payload = {'fixed_summary': fixed, 'full_joint_summary': baseline, 'oracle_diagnostic_summary': oracle,
               'paired_fixed_changes': paired, 'confirmation_gate': gate, 'gate_pass': all(gate.values()),
               'verified_hybrid_artifacts': checked, 'runs': runs}
    (OUT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Hybrid-V2: three-seed results', '',
             'BiDA stem + Full Joint BN + existing A1 query tokenizer package; depth 3,',
             'dropout 0.1, source self CE, 200 epochs. Seeds 2100/2101/2102.',
             'Mean ± sample SD. No checkpoint selection or tuning from target labels.', '',
             '## Primary: fixed epoch 200', '', '| Model | OA | AA | Kappa | Class-6 collapse |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label, summary in (('Full Joint + BiDA tokenizer', baseline), ('Hybrid-V2', fixed)):
        cells = [f"{summary[key]['mean']:.2f} ± {summary[key]['sample_sd']:.2f}" for key in ('oa_percent', 'aa_percent')]
        k = summary['kappa']
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {k['mean']:.3f} ± {k['sample_sd']:.3f} | {summary['class6_collapse_count']}/3 |")
    lines += ['', f"Predeclared gate passes: {all(gate.values())}. Mean fixed OA change: {fixed['oa_percent']['mean'] - baseline['oa_percent']['mean']:+.2f} pp; mean AA change: {fixed['aa_percent']['mean'] - baseline['aa_percent']['mean']:+.2f} pp.",
              '', '| Seed | OA | AA | Kappa | OA vs control | AA vs control | Class-6 share | Max class share | Recall coverage | Source-val OA |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        run = runs[str(seed)]
        row, delta = run['target_at_final'], paired[str(seed)]
        lines.append(f"| {seed} | {row['oa_percent']:.2f} | {row['aa_percent']:.2f} | {row['kappa']:.3f} | {delta['oa_points']:+.2f} | {delta['aa_points']:+.2f} | {100*row['prediction_shares'][5]:.2f}% | {100*run['max_class_share_final']:.2f}% | {run['recall_coverage_final']}/7 | {run['source_val_final']['oa_percent']:.2f} |")
    lines += ['', '## Secondary: target-oracle diagnostics only', '',
              'Highest target OA among epochs 10,20,...,200; earliest epoch wins ties.',
              'Not a target-blind estimate and not the formal selected checkpoint.', '',
              '| Seed | Best epoch | OA | Corresponding AA | Corresponding Kappa | OA minus fixed |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        run = runs[str(seed)]
        row = run['target_oracle']['metrics']
        lines.append(f"| {seed} | {run['target_oracle']['best_epoch']} | {row['oa_percent']:.2f} | {row['aa_percent']:.2f} | {row['kappa']:.3f} | {row['oa_percent']-run['target_at_final']['oa_percent']:+.2f} |")
    lines += ['', f"Oracle OA: {oracle['oa_percent']['mean']:.2f} ± {oracle['oa_percent']['sample_sd']:.2f}%. This is a label-selected diagnostic, not a guaranteed potential ceiling.",
              '', '## Fixed per-class recall (%)', '', '| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        lines.append(f'| {seed} | ' + ' | '.join(f'{x:.2f}' for x in runs[str(seed)]['target_at_final']['per_class_percent']) + ' |')
    lines.append('| Mean ± SD | ' + ' | '.join(f'{m:.2f} ± {s:.2f}' for m, s in zip(fixed['recall_mean'], fixed['recall_sample_sd'])) + ' |')
    lines += ['', '## Fixed prediction distribution', '', '| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        row = runs[str(seed)]['target_at_final']
        lines.append(f'| {seed} | ' + ' | '.join(f'{n} ({100*s:.2f}%)' for n, s in zip(row['prediction_counts'], row['prediction_shares'])) + ' |')
    lines += ['', '## Frozen trajectory collapse', '', '| Seed | Class-6 collapse grid points | Any-class ≥95% grid points |', '| --- | ---: | ---: |']
    for seed in SEEDS:
        trajectory = runs[str(seed)]['target_trajectory']
        c6 = sum(row['metrics']['prediction_shares'][5] >= .95 for row in trajectory)
        anyclass = sum(max(row['metrics']['prediction_shares']) >= .95 for row in trajectory)
        lines.append(f'| {seed} | {c6}/20 | {anyclass}/20 |')
    lines += ['', '## Interpretation', '',
              'This experiment does not confirm complementarity: all three fixed-200 OA endpoints are below the paired Full Joint + BiDA-tokenizer controls, and seed 2101 collapses.',
              'Seed 2101 first crosses the class-6-collapse threshold at epoch 60 and ends with 97.90% class-6 predictions and zero recall for classes 1, 2 and 7. Its oracle OA is only 70.44%, so late checkpoint selection alone does not explain the failure.',
              'Seed 2102 remains below its paired control throughout the sampled trajectory, even at its own target-oracle endpoint. All three final source validation accuracies are 100%, again showing that source validation does not guarantee target coverage.',
              'Full Joint stabilization is established for the tested BiDA tokenizer, not universally for every tokenizer. Retain Full Joint + BiDA tokenizer as the reference; do not promote Hybrid-V2 to the main backbone.',
              'The specific cause of query-package instability is not identified by this combination experiment. No additional architecture or hyperparameter search was performed.',
              '', '## Reproducibility and scope', '',
              f'Verified {checked} Hybrid checkpoint/prediction hashes plus all Hybrid and paired control manifests.',
              'All three original BiDA initial hashes and existing Hybrid transplant hashes match. Full Joint installation changes no initial tensor.',
              'All 20 target predictions were frozen before target metrics. Full per-epoch metrics, confusion matrices and entropy are in results/hybrid_v2/summary.json.',
              'The intervention includes query, positional encoding, cross-attention, FFN and changed parameter count. It does not isolate learnable queries alone.',
              'Three optimization seeds on one scene pair do not establish cross-dataset stability. Zero class-6 collapse does not imply stable minority-class recall.']
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for i, seed in enumerate(SEEDS):
        for j, field in enumerate(('oa_percent', 'aa_percent', 'prediction_shares')):
            axis = axes[i, j]
            for label, run in (('Full Joint + BiDA', controls[str(seed)]), ('Hybrid-V2', runs[str(seed)])):
                trajectory = run['target_trajectory']
                values = [100*row['metrics'][field][5] if field == 'prediction_shares' else row['metrics'][field] for row in trajectory]
                axis.plot([row['epoch'] for row in trajectory], values, label=label)
            axis.set_title(f"{seed}: {field if field != 'prediction_shares' else 'Class-6 share (%)'}")
            axis.set_ylim(0, 103)
            axis.grid(alpha=.25)
            if field == 'prediction_shares':
                axis.axhline(95, color='red', linestyle='--', linewidth=.8)
            if i == 2:
                axis.set_xlabel('Epoch')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'trajectories.png', dpi=150)
    plt.close(fig)
    print(json.dumps({key: payload[key] for key in ('fixed_summary', 'full_joint_summary', 'oracle_diagnostic_summary', 'paired_fixed_changes', 'confirmation_gate')}, indent=2))

if __name__ == '__main__':
    main()
