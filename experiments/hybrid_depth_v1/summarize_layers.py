"""Report stage concentration; auxiliary head lens is not early-exit accuracy."""
import json
from pathlib import Path
import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hybrid_depth_v1/layer_probes'

def main():
    payloads = [json.loads(path.read_text()) for path in sorted(OUT.glob('*.json'))]
    assert len({x['sample_sha256'] for x in payloads}) == 1
    assert {(p['arm'],p['seed']) for p in payloads} == {(a,s) for a in ('bida3','query1','query3') for s in (2100,2101,2102)}
    for payload in payloads:
        assert [r['epoch'] for r in payload['rows']] == list(range(10,201,10))
        assert payload['layers_script_sha256'] == train.helper.file_hash(Path(__file__).with_name('layers.py'))
        assert payload['probe_script_sha256'] == train.helper.file_hash(Path(__file__).with_name('probe.py'))
        assert payload['protocol_sha256'] == train.helper.file_hash(Path(__file__).with_name('PROTOCOL.md'))
    lines = ['# Hybrid depth V1: residual-stage probes', '',
             'Fixed2048 source/target samples identical to the previous probe. Eval mode,',
             'no backward/updates. Hook logits were verified unchanged; state hashes unchanged.',
             'Cosines are to each model\'s local final classifier w6; early stages are not',
             'in the final LayerNorm coordinate system. The auxiliary margin applies the',
             'frozen final norm/head to early vectors; it is not a trained early-exit head.',
             'Initial CLS has concentration1 by construction: this is not learned collapse.', '',
             '## Seed2101 epoch200 stage comparison', '',
             '| Arm | Domain | Stage | Norm | Direction concentration | Raw cos(w6) | Auxiliary margin6 | Auxiliary class6 share |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for payload in payloads:
        if payload['seed'] != 2101:
            continue
        row = next(x for x in payload['rows'] if x['epoch'] == 200)
        for domain, stages in row['domains'].items():
            for key, r in stages.items():
                lines.append(f"| {payload['arm']} | {domain} | {key} | {r['mean_norm']:.3f} | {r['direction_concentration']:.3f} | {r['cos_w6_mean']:.3f} | {r['aux_final_norm_head_margin6']:.3f} | {100*r['aux_final_norm_head_class6_share']:.2f}% |")
    lines += ['', '## Seed2101 target trajectory by residual stage', '',
              '| Arm | Epoch | Stage | Concentration | Raw cos(w6) | Auxiliary margin6 |', '| --- | ---: | --- | ---: | ---: | ---: |']
    for payload in payloads:
        if payload['seed'] != 2101:
            continue
        for row in payload['rows']:
            for key, r in row['domains']['target'].items():
                if key == 'tokenizer_mean' or key.endswith(('_after_attention', '_after_mlp')) or key == 'final_cls':
                    lines.append(f"| {payload['arm']} | {row['epoch']} | {key} | {r['direction_concentration']:.3f} | {r['cos_w6_mean']:.3f} | {r['aux_final_norm_head_margin6']:.3f} |")
    lines += ['', '## Interpretation of the seed2101 boundary probes', '',
              'Depth3 epoch200: the token-mean concentration is similar for Query and BiDA (0.870 versus 0.854), but block1 after-attention CLS differs sharply (0.908 versus 0.573). After block1 MLP the Query concentration is already 0.922, compared with BiDA 0.596. The pronounced CLS concentration is therefore present in block1, not first appearing in block3.',
              'For the depth3 Query model, raw local cosine to w6 is 0.360 after block1 attention, 0.480 after block1 MLP, 0.568 after block2 attention, 0.646 after block2 MLP, 0.688 after block3 attention, and 0.690 after block3 MLP. Block2 further strengthens alignment; there is no unique single guilty sublayer.',
              'The independently trained depth1 model also shows high target concentration: block1 after attention 0.889 and after MLP 0.882. MLP slightly reduces concentration here while increasing cosine to w6 from 0.415 to 0.594. Concentration and class-specific alignment must be distinguished; not every MLP monotonically concentrates directions.',
              'Source depth1 final concentration is only 0.107, compared with target 0.882, so the signature is domain-specific. Final target concentration is already high at epoch10 (0.835), rises to 0.884 by epoch60, and is 0.882 at epoch200. Depth3 final target concentration is 0.920 at epoch200.',
              'These comparisons show residual-stage signatures in the learned system, not proof that changing an attention/MLP component will repair it. Token-mean and CLS are distinct vectors; initial CLS is constant. Direct cosine to the final head is a descriptive coordinate lens, not a validated early-exit predictor.',
              '', 'These are observational probes, not causal layer lesions. Depth1 retraining',
              'changes all learned weights and BN statistics over optimization. Attribution',
              'to a particular attention or MLP component requires a separate intervention.']
    Path(__file__).with_name('LAYERS.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'summary.json').write_text(json.dumps({'probes': payloads, 'verified_checkpoint_count': 180},indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for payload in payloads:
        if payload['seed'] != 2101:
            continue
        stages = [k for k in payload['rows'][0]['domains']['target'] if k.endswith(('_after_attention','_after_mlp'))]
        for key in stages:
            rows = payload['rows']
            for axis, metric in zip(axes, ('direction_concentration','cos_w6_mean','aux_final_norm_head_margin6')):
                axis.plot([r['epoch'] for r in rows],[r['domains']['target'][key][metric] for r in rows],label=payload['arm']+'/'+key)
                axis.set_title(metric)
                axis.grid(alpha=.2)
                axis.set_xlabel('Epoch')
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(OUT / 'seed2101_stages.png',dpi=160)
    plt.close(fig)
    print('Layer probe reports:', [(p['arm'],p['seed']) for p in payloads])

if __name__ == '__main__':
    main()
