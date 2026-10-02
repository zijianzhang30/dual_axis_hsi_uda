"""Summarize reconstructed checkpoint probes, without causal claims."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/joint_tokenizer_probe'

def main():
    runs = {seed: json.loads((OUT / f'seed_{seed}.json').read_text()) for seed in (2100, 2101, 2102)}
    assert len({run['sample_sha256'] for run in runs.values()}) == 1
    lines = ['# Joint-BN tokenizer retrospective diagnosis', '',
             'Identical label-unstratified fixed probes: 2048 patches per domain, batch 128.',
             'Frozen checkpoint weights only. Eval uses saved buffers; joint-train probes',
             'use current paired moments, dropout disabled, no gradients or updates.',
             'Every state restoration was hash-checked. These are reconstructed probes,',
             'not recorded historical training activations.', '',
             'Direction concentration is the norm of the mean unit vector; higher means',
             'more similar directions across samples. Cosine values are model-local.', '',
             '## Fixed-200 eval target probes', '',
             '| Seed | Tokenizer | BN2 variance | Token norm | Token/feature RMS ratio | Pre-LN CLS norm | z norm | z concentration | Cos(z,w6) | Class-6 margin |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed, run in runs.items():
        for arm in ('bida', 'query'):
            row = next(x for x in run['rows'] if x['arm'] == arm and x['epoch'] == 200)
            r = row['modes']['eval']['target']
            lines.append(f"| {seed} | {arm} | {r['bn2d']['mean_channel_variance']:.3f} | {r['tokens']['mean_norm']:.3f} | {r['token_to_bn2d_rms_ratio']:.3f} | {r['pre_norm_cls']['mean_norm']:.3f} | {r['z']['mean_norm']:.3f} | {r['z']['direction_concentration']:.3f} | {r['classifier_cosine_mean'][5]:.3f} | {r['class6_margin_mean']:.3f} |")
    lines += ['', '## Seed 2101: eval trajectory', '',
              '| Tokenizer | Epoch | BN2 variance | Token norm | Pre-LN CLS norm | z concentration | Cos(z,w6) | Class-6 margin | Probe class-6 share | Full-target class-6 share |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in runs[2101]['rows']:
        r = row['modes']['eval']['target']
        lines.append(f"| {row['arm']} | {row['epoch']} | {r['bn2d']['mean_channel_variance']:.3f} | {r['tokens']['mean_norm']:.3f} | {r['pre_norm_cls']['mean_norm']:.3f} | {r['z']['direction_concentration']:.3f} | {r['classifier_cosine_mean'][5]:.3f} | {r['class6_margin_mean']:.3f} | {100*r['prediction_shares'][5]:.2f}% | {100*row['full_target_metrics']['prediction_shares'][5]:.2f}% |")
    lines += ['', '## Seed 2101: fixed-200 source/target and mode comparison', '',
              '| Tokenizer | Mode | Domain | BN2 abs mean | BN2 variance | Token norm | z concentration | Cos(z,w6) | Class-6 margin | Class-6 share |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in runs[2101]['rows']:
        if row['epoch'] != 200:
            continue
        for mode in ('eval', 'joint_train'):
            for domain in ('source', 'target'):
                r = row['modes'][mode][domain]
                lines.append(f"| {row['arm']} | {mode} | {domain} | {r['bn2d']['mean_channel_abs_mean']:.3f} | {r['bn2d']['mean_channel_variance']:.3f} | {r['tokens']['mean_norm']:.3f} | {r['z']['direction_concentration']:.3f} | {r['classifier_cosine_mean'][5]:.3f} | {r['class6_margin_mean']:.3f} | {100*r['prediction_shares'][5]:.2f}% |")
    lines += ['', '## Evidence-backed interpretation', '',
              'Seed 2101 does not show token-norm explosion: at epoch 200 Query token norm is 2.847 versus BiDA 4.403. BN2 target variance is 0.671 versus 0.719; classifier-input z norm is 8.109 versus 8.151.',
              'The stronger signature is target direction concentration after the downstream transformer. Token direction concentration is relatively close (Query 0.868, BiDA 0.851), whereas pre-final-LayerNorm CLS concentration is 0.920 versus 0.618. The final LayerNorm preserves this difference; it does not create it at that boundary.',
              'Target cosine to the local class-6 classifier weight rises to 0.693 versus 0.447. Class-6 weight norm/bias do not explode: Query 1.333/0.00908 versus BiDA 1.368/0.00952. Mean target class-6 logits are 7.50 versus 5.00; margins over the best other logit are 6.75 versus 2.18.',
              'The query classifier representation remains diverse on source (concentration 0.143) but concentrates on target (0.920). This is domain-specific representational collapse, not a globally broken classifier.',
              'Query target directional concentration is already elevated at epoch 10 (0.859), reverses at epoch 20, then remains high from epoch 30 (0.899) with margin 5.18. Full-target class-6 share first crosses the predeclared 95% collapse threshold at epoch 60. Therefore there is no single monotonically increasing early onset; the sustained separation starts around the sampled epoch-30 checkpoint.',
              'Current joint-training moments do not restore diversity at epoch 200: Query target concentration 0.922, margin 6.79, class-6 share 98.10%; eval share is 97.56% on the same sample. This is not solely saved running-buffer mismatch.',
              'The observations localize the pronounced concentration change somewhere between tokenizer outputs and pre-final-LayerNorm CLS features in the learned tokenizer-plus-transformer system. They do not prove that the query reader alone rotates features or that a residual/normalization change will repair it. Per-block or controlled-intervention evidence would be needed for that claim.',
              'Retain Full Joint + BiDA tokenizer as the reference. No rescue tuning or query normalization changes were made.',
              '', '## Limitations', '',
              'BN statistics are per-channel averages of pre-ReLU outputs. Token norms are',
              'after different tokenizer packages; their absolute scales are not intrinsically',
              'comparable as a defect. The token/feature ratio is descriptive, not a causal gain.',
              'Train-mode probes disable dropout and omit historical augmentation; the sample',
              'is not class-stratified. We do not use GT labels for grouping or sampling.',
              'Observed scale/direction changes do not identify the tokenizer as the sole cause:',
              'the stem, normalization affine parameters, transformer and classifier co-adapt.',
              'No architecture changes or normalization fixes were made. Full JSON includes',
              'per-channel vectors, classifier norms/biases, all source/target modes and seeds.']
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'runs': {str(k): v for k, v in runs.items()}, 'verified_checkpoint_count': 120}, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = [('bn2d', 'mean_channel_variance', 'BN2 variance'), ('tokens', 'mean_norm', 'Token norm'),
              ('pre_norm_cls', 'mean_norm', 'Pre-LN CLS norm'), ('z', 'direction_concentration', 'z concentration'),
              (None, 'class6_margin_mean', 'Class-6 margin'), (None, 'classifier_cosine_mean', 'Cos(z,w6)')]
    fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharex=True)
    for i, seed in enumerate(runs):
        for j, (nested, key, title) in enumerate(fields):
            for arm in ('bida', 'query'):
                rows = [x for x in runs[seed]['rows'] if x['arm'] == arm]
                for mode, style in (('eval', '-'), ('joint_train', '--')):
                    values = []
                    for row in rows:
                        r = row['modes'][mode]['target']
                        value = r[nested][key] if nested else r[key]
                        values.append(value[5] if key == 'classifier_cosine_mean' else value)
                    axes[i, j].plot([x['epoch'] for x in rows], values, style, color='tab:blue' if arm == 'bida' else 'tab:orange', label=f'{arm}/{mode}')
            axes[i, j].set_title(f'{seed}: {title}')
            axes[i, j].grid(alpha=.25)
            if i == 2:
                axes[i, j].set_xlabel('Epoch')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'trajectories.png', dpi=130)
    plt.close(fig)
    print('\n'.join(lines[:19]))

if __name__ == '__main__':
    main()
