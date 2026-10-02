"""Fixed endpoints, separately labeled oracle, and query-pipeline controls."""
import json
from pathlib import Path
import numpy as np
import train

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/query_class_readout_v1'
SEEDS = (2100,2101,2102)
ARMS = ('pipeline','query1','query3','bida3')

def folder(arm,seed):
    if arm=='bida3':
        return ROOT / f'results/bn_stabilization_v1/fixed_mixture_{seed}'
    directory = {'pipeline':'query_class_readout_v1','query1':'hybrid_depth_v1','query3':'hybrid_v2'}[arm]
    return ROOT / f'results/{directory}/formal_{seed}'

def aggregate(rows):
    summary = {'class6_collapse_count':sum(x['prediction_shares'][5]>=.95 for x in rows)}
    for key in ('oa_percent','aa_percent','kappa'):
        a = np.array([x[key] for x in rows])
        summary[key] = {'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1))}
    return summary

def main():
    runs,summaries,checked={},{},0
    for arm in ARMS:
        rows=[]
        for seed in SEEDS:
            path=folder(arm,seed)
            result=json.loads((path/'results.json').read_text())
            mp=path/'frozen_manifest.json'
            assert train.helper.file_hash(mp)==result['frozen_manifest_sha256']
            manifest=json.loads(mp.read_text())
            for group in ('checkpoints','predictions'):
                assert [x['epoch'] for x in manifest[group]]==list(range(10,201,10))
                for x in manifest[group]:
                    assert train.helper.file_hash(x['path'])==x['sha256']
                    checked+=1
            if arm=='pipeline':
                config=manifest['config']
                assert config['depth']==0 and config['bn_momentum_per_step']==.19
                for filename,key in (('PROTOCOL.md','protocol_sha256'),('train.py','train_script_sha256'),('pipeline.py','pipeline_script_sha256')):
                    assert train.helper.file_hash(Path(__file__).with_name(filename))==config[key]
                assert train.helper.file_hash(ROOT/'model.py')==config['a1_script_sha256']
                assert train.helper.file_hash(ROOT/'experiments/bn_stabilization_v1/normalization.py')==config['joint_script_sha256']
                history=[json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()]
                assert len(history)==200
                result['source_val_final']=history[-1]['source_val']
                result['trainable_parameters']=config['trainable_parameters']
                best=max(result['target_trajectory'],key=lambda x:x['metrics']['oa_percent'])
                assert result['target_oracle']=={'best_epoch':best['epoch'],'metrics':best['metrics']}
            runs[f'{arm}_{seed}']=result
            rows.append(result['target_at_final'])
        summaries[arm]=aggregate(rows)
    oracle=aggregate([runs[f'pipeline_{s}']['target_oracle']['metrics'] for s in SEEDS])
    changes={arm:{key:summaries['pipeline'][key]['mean']-summaries[arm][key]['mean'] for key in ('oa_percent','aa_percent','kappa')} for arm in ARMS[1:]}
    payload={'summaries':summaries,'oracle_diagnostic':oracle,'mean_pipeline_minus_controls':changes,'verified_artifacts':checked,'runs':runs}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Query tokenizer + A1 class-query readout (no BiDA blocks)','',
           'Same Full Joint BN; three seeds2100/2101/2102, 200 epochs. Mean ± sample SD.',
           'Pipeline changes readout as well as depth; it is not an isolated block-removal control.','',
           '## Fixed epoch200 primary','', '| Arm | OA | AA | Kappa | Class6 collapse |','| --- | ---: | ---: | ---: | ---: |']
    for arm,s in summaries.items():
        cells=[f"{s[k]['mean']:.2f} ± {s[k]['sample_sd']:.2f}" for k in ('oa_percent','aa_percent')]
        k=s['kappa']
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {k['mean']:.3f} ± {k['sample_sd']:.3f} | {s['class6_collapse_count']}/3 |")
    lines+=['','| Seed | OA | AA | Kappa | Class6 share | Max class share | Recall coverage | Source-val OA |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        run=runs[f'pipeline_{seed}'];r=run['target_at_final']
        lines.append(f"| {seed} | {r['oa_percent']:.2f} | {r['aa_percent']:.2f} | {r['kappa']:.3f} | {100*r['prediction_shares'][5]:.2f}% | {100*max(r['prediction_shares']):.2f}% | {sum(x>0 for x in r['per_class_percent'])}/7 | {run['source_val_final']['oa_percent']:.2f} |")
    lines+=['','## Oracle diagnostic only','', '| Seed | Best epoch | OA | Associated AA | Associated Kappa |','| --- | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        o=runs[f'pipeline_{seed}']['target_oracle'];r=o['metrics']
        lines.append(f"| {seed} | {o['best_epoch']} | {r['oa_percent']:.2f} | {r['aa_percent']:.2f} | {r['kappa']:.3f} |")
    lines+=['',f"Oracle OA {oracle['oa_percent']['mean']:.2f} ± {oracle['oa_percent']['sample_sd']:.2f}%. Label-selected, not a formal checkpoint or guaranteed upper bound.",'',
            '## Fixed per-class recall','', '| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        r=runs[f'pipeline_{seed}']['target_at_final']
        lines.append(f'| {seed} | '+' | '.join(f'{x:.2f}' for x in r['per_class_percent'])+' |')
    lines+=['','## Fixed prediction distribution','', '| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        r=runs[f'pipeline_{seed}']['target_at_final']
        lines.append(f'| {seed} | '+' | '.join(f'{n} ({100*s:.2f}%)' for n,s in zip(r['prediction_counts'],r['prediction_shares']))+' |')
    lines+=['','## Trajectory coverage','', '| Seed | Class6 collapse grid points | Any-class ≥95% grid points |','| --- | ---: | ---: |']
    for seed in SEEDS:
        t=runs[f'pipeline_{seed}']['target_trajectory']
        lines.append(f"| {seed} | {sum(x['metrics']['prediction_shares'][5]>=.95 for x in t)}/20 | {sum(max(x['metrics']['prediction_shares'])>=.95 for x in t)}/20 |")
    lines+=['','## Mean pipeline minus controls','']
    for arm,d in changes.items():
        lines.append(f"- {arm}: OA {d['oa_percent']:+.2f} pp, AA {d['aa_percent']:+.2f} pp.")
    probes={}
    for seed in SEEDS:
        probe=json.loads((OUT/f'probes/seed_{seed}.json').read_text())
        assert probe['probe_script_sha256']==train.helper.file_hash(Path(__file__).with_name('probe.py'))
        assert [r['epoch'] for r in probe['rows']]==list(range(10,201,10))
        probes[str(seed)]=probe
    assert len({p['sample_sha256'] for p in probes.values()})==1
    payload['target_angular_probes']=probes
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines+=['','## Fixed target angular probe (same2048 samples as prior diagnosis)','',
            '| Seed | Stage | Norm | Direction concentration | Cos(w6) |','| --- | --- | ---: | ---: | ---: |']
    for seed in SEEDS:
        p=probes[str(seed)]['rows'][-1]['target']
        for stage in ('spatial_tokens','class_reader','classifier_z'):
            r=p[stage]
            lines.append(f"| {seed} | {stage} | {r['mean_norm']:.3f} | {r['direction_concentration']:.3f} | {r['classifier_cosine_mean'][5]:.3f} |")
    lines+=['','Stage cosines are model-local descriptive lenses. Pre-final-normalizer vectors are not in identical coordinates to final z; no causal attribution from cosine alone.',
            '', '## Interpretation', '',
            'The no-BiDA-block pipeline can achieve high fixed OA in seeds2100/2102 (80.46/81.42%), but does not reproduce the balanced three-seed stability of Full Joint + BiDA tokenizer. Its mean fixed OA is 0.69 pp lower, AA is 9.53 pp lower, and both OA/AA sample SDs are larger.',
            'None of the60 frozen grid points cross the predefined class6-collapse threshold. Nevertheless seed2101 has 83.41% class6 predictions, zero class7 recall, and only 47.24% AA. Avoiding a95% threshold is not equivalent to recovering category coverage.',
            'On the same fixed2048 target probe, seed2101 classifier z norm is 8.114, directional concentration is 0.867, cosine to w6 is 0.636 and class6 margin is 4.757. Strong target directional/class bias persists without any BiDA refinement block.',
            'Therefore BiDA refinement is not required for the observed strong majority-class directional bias in a trained query pipeline. Depth3 worsens the earlier Hybrid controls, but removing it and replacing the readout does not solve the whole problem.',
            'This experiment does not establish that the spatial query tokenizer is intrinsically wrong or that the original interface is the sole cause: the new A1 class reader, retained classifier and stem also co-adapt during training. Do not promote this pipeline to a stable primary backbone; keep Full Joint + BiDA tokenizer as reference.',
            '', '## Scope and checks','',
            f'Verified {checked} checkpoint/prediction hashes and 12 manifests; all20 predictions per new run frozen before target metrics.',
            'Initial retained stem/tokenizer/final-normalizer/classifier tensors match the existing per-seed Hybrid initialization. New class query/reader initialization and removal of blocks alter training RNG consumption.',
            f"New pipeline has {runs['pipeline_2100']['trainable_parameters']:,} trainable parameters. Final classifier retains BiDA initial tensors in A1-compatible LayerNorm/linear structure.",
            'No new loss, BN rule, residual scale, learning-rate or checkpoint tuning. Three seeds on one transfer pair do not establish general stability.',
            'Success/failure concerns the whole tokenizer-plus-class-reader pipeline, not the spatial-query tokenizer alone. Readout and parameter-count changes confound a pure depth attribution.']
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True)
    for i,seed in enumerate(SEEDS):
        for j,field in enumerate(('oa_percent','aa_percent','prediction_shares')):
            for arm in ARMS:
                t=runs[f'{arm}_{seed}']['target_trajectory']
                values=[100*x['metrics'][field][5] if field=='prediction_shares' else x['metrics'][field] for x in t]
                axes[i,j].plot([x['epoch'] for x in t],values,label=arm)
            axes[i,j].set_title(f'{seed}: {field}');axes[i,j].set_ylim(0,103);axes[i,j].grid(alpha=.2)
            if field=='prediction_shares':axes[i,j].axhline(95,color='red',linestyle='--',linewidth=.8)
            if i==2:axes[i,j].set_xlabel('Epoch')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'trajectories.png',dpi=150);plt.close(fig)
    print(json.dumps({'summaries':summaries,'oracle':oracle,'changes':changes},indent=2))

if __name__=='__main__':main()
