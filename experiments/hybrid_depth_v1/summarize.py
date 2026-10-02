"""Report depth controls separately from retrospective stage diagnostics."""
import json
from pathlib import Path
import numpy as np
import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hybrid_depth_v1'
SEEDS = (2100, 2101, 2102)

def main():
    runs, summaries, checked = {}, {}, 0
    for arm in ('query1', 'query3', 'bida3'):
        rows = []
        for seed in SEEDS:
            if arm == 'bida3':
                folder = ROOT / f'results/bn_stabilization_v1/fixed_mixture_{seed}'
            else:
                folder = ROOT / ('results/hybrid_depth_v1' if arm == 'query1' else 'results/hybrid_v2') / f'formal_{seed}'
            result = json.loads((folder / 'results.json').read_text())
            manifest_path = folder / 'frozen_manifest.json'
            assert train.helper.file_hash(manifest_path) == result['frozen_manifest_sha256']
            manifest = json.loads(manifest_path.read_text())
            for group in ('checkpoints', 'predictions'):
                assert [x['epoch'] for x in manifest[group]] == list(range(10, 201, 10))
                for item in manifest[group]:
                    assert train.helper.file_hash(item['path']) == item['sha256']
                    checked += 1
            if arm == 'query1':
                config = manifest['config']
                assert config['depth'] == 1 and config['bn_momentum_per_step'] == .19
                assert train.helper.file_hash(Path(__file__).with_name('PROTOCOL.md')) == config['protocol_sha256']
                assert train.helper.file_hash(Path(__file__).with_name('train.py')) == config['train_script_sha256']
                history = [json.loads(x) for x in (folder / 'history.jsonl').read_text().splitlines()]
                assert len(history) == 200
                result['source_val_final'] = history[-1]['source_val']
            runs[f'{arm}_{seed}'] = result
            rows.append(result['target_at_final'])
        summary = {'collapse_count': sum(x['prediction_shares'][5] >= .95 for x in rows)}
        for key in ('oa_percent', 'aa_percent', 'kappa'):
            values = np.array([x[key] for x in rows])
            summary[key] = {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))}
        summaries[arm] = summary
    paired = {str(seed): {'oa': runs[f'query1_{seed}']['target_at_final']['oa_percent'] - runs[f'query3_{seed}']['target_at_final']['oa_percent'],
                          'aa': runs[f'query1_{seed}']['target_at_final']['aa_percent'] - runs[f'query3_{seed}']['target_at_final']['aa_percent']} for seed in SEEDS}
    payload = {'summaries': summaries, 'paired_depth1_minus_depth3': paired, 'verified_artifacts': checked, 'runs': runs}
    (OUT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Hybrid depth V1: fixed endpoints', '', 'Full Joint BN unchanged. Query package depth1/3 and BiDA-tokenizer depth3.',
             'Seeds 2100/2101/2102; epoch200 primary. Mean ± sample SD.', '',
             '| Arm | OA | AA | Kappa | Class6 collapse |', '| --- | ---: | ---: | ---: | ---: |']
    for arm, summary in summaries.items():
        cells = [f"{summary[k]['mean']:.2f} ± {summary[k]['sample_sd']:.2f}" for k in ('oa_percent', 'aa_percent')]
        k = summary['kappa']
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {k['mean']:.3f} ± {k['sample_sd']:.3f} | {summary['collapse_count']}/3 |")
    lines += ['', '| Arm | Seed | OA | AA | Kappa | Class6 share | Recall coverage | Oracle OA | Oracle epoch |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('query1', 'query3'):
        for seed in SEEDS:
            run = runs[f'{arm}_{seed}']
            row, oracle = run['target_at_final'], run['target_oracle']
            lines.append(f"| {arm} | {seed} | {row['oa_percent']:.2f} | {row['aa_percent']:.2f} | {row['kappa']:.3f} | {100*row['prediction_shares'][5]:.2f}% | {sum(x>0 for x in row['per_class_percent'])}/7 | {oracle['metrics']['oa_percent']:.2f} | {oracle['best_epoch']} |")
    lines += ['', '## Depth1 fixed per-class recall', '', '| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        row = runs[f'query1_{seed}']['target_at_final']
        lines.append(f'| {seed} | ' + ' | '.join(f'{x:.2f}' for x in row['per_class_percent']) + ' |')
    lines += ['', '## Depth1 prediction distribution', '', '| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        row = runs[f'query1_{seed}']['target_at_final']
        lines.append(f'| {seed} | ' + ' | '.join(f'{n} ({100*s:.2f}%)' for n,s in zip(row['prediction_counts'],row['prediction_shares'])) + ' |')
    lines += ['', '## Paired depth1 minus depth3 changes', '']
    for seed, row in paired.items():
        lines.append(f"- {seed}: OA {row['oa']:+.2f} pp, AA {row['aa']:+.2f} pp.")
    lines += ['', '## Interpretation', '',
              'Depth1 improves mean fixed OA by 7.64 pp and AA by 12.64 pp versus Query depth3; OA sample SD falls from 7.78 to 2.62. None of its 60 frozen grid checkpoints cross the class6-collapse threshold.',
              'This is a meaningful depth-related rescue, not full stability: seed2101 still assigns 83.03% of predictions to class6 and has class7 recall zero. Mean OA/AA remain 1.93/7.07 pp below Full Joint + BiDA-tokenizer depth3, with larger across-seed AA variation.',
              'Layer probes show elevated target CLS concentration in block1 for both trained depths, and further class6 alignment in block2 of the depth3 model. The results support depth-dependent worsening of a target representational bias; they do not establish that block1 alone is a sufficient cause or that an interface normalization patch will fix it.',
              'Keep Full Joint + BiDA tokenizer as the main reference. Query depth1 remains an exploratory control, not a promoted backbone.',
              '', '## Scope and reproducibility', '',
              f'Verified {checked} checkpoint/prediction SHA256 entries and nine manifests.',
              'All initial retained tensors match the full initialized Hybrid. Only blocks2/3 are removed; Full Joint BN and classifier/readout are unchanged.',
              'Fixed200 results remain formal; oracle OA is target-selected and diagnostic only. All20 predictions were frozen before post-hoc target metrics.',
              'Depth0 is deferred with user agreement. Three seeds/one transfer pair and removal of blocks do not uniquely isolate an interface defect. Dropout RNG consumption also changes with depth.',
              'Stage probes are in LAYERS.md. No decorrelation loss, cosine classifier, BN change or rescue tuning was introduced.']
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12,9), sharex=True)
    for i, seed in enumerate(SEEDS):
        for j, field in enumerate(('oa_percent','aa_percent','prediction_shares')):
            for arm in ('query1','query3','bida3'):
                trajectory = runs[f'{arm}_{seed}']['target_trajectory']
                values = [100*x['metrics'][field][5] if field=='prediction_shares' else x['metrics'][field] for x in trajectory]
                axes[i,j].plot([x['epoch'] for x in trajectory], values, label=arm)
            axes[i,j].set_title(f'{seed}: {field}')
            axes[i,j].set_ylim(0,103)
            axes[i,j].grid(alpha=.25)
            if field=='prediction_shares':
                axes[i,j].axhline(95,color='red',linestyle='--',linewidth=.8)
            if i==2:
                axes[i,j].set_xlabel('Epoch')
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'trajectories.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'summaries': summaries, 'paired': paired}, indent=2))

if __name__ == '__main__':
    main()
