"""Single stress-seed factorial contrasts; not a general causal attribution."""
import json
from pathlib import Path
import train

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tokenizer_readout_v1'
ARMS=('bida_mean','bida_class','query_mean','query_class')

def folder(arm):
    return ROOT/'results/query_class_readout_v1/formal_2101' if arm=='query_class' else OUT/f'{arm}_2101'

def main():
    runs,probes,checked={},{},0
    for arm in ARMS:
        path=folder(arm);result=json.loads((path/'results.json').read_text())
        mp=path/'frozen_manifest.json';assert train.helper.file_hash(mp)==result['frozen_manifest_sha256']
        manifest=json.loads(mp.read_text());config=manifest['config']
        assert config['seed']==2101 and config['depth']==0 and config['bn_momentum_per_step']==.19
        for group in ('checkpoints','predictions'):
            assert [x['epoch'] for x in manifest[group]]==list(range(10,201,10))
            for x in manifest[group]:assert train.helper.file_hash(x['path'])==x['sha256'];checked+=1
        if arm!='query_class':
            for file,key in (('PROTOCOL.md','protocol_sha256'),('train.py','train_script_sha256'),('pipeline.py','pipeline_script_sha256')):
                assert train.helper.file_hash(Path(__file__).with_name(file))==config[key]
            reference=json.loads((folder('query_class')/'config.json').read_text())
            assert config['initial_common_query_class_sha256']==reference['initial_pipeline_sha256']
        history=[json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()];assert len(history)==200
        result['source_val_final']=history[-1]['source_val'];result['parameters']=config['trainable_parameters'];runs[arm]=result
        probe=json.loads((OUT/f'probes/{arm}.json').read_text())
        assert probe['script_sha256']==train.helper.file_hash(Path(__file__).with_name('probe.py'))
        assert [r['epoch'] for r in probe['rows']]==list(range(10,201,10));probes[arm]=probe
    assert len({p['sample_sha256'] for p in probes.values()})==1
    contrasts={}
    for key in ('oa_percent','aa_percent','kappa'):
        v={a:r['target_at_final'][key] for a,r in runs.items()}
        contrasts[key]={'query_minus_bida_under_class':v['query_class']-v['bida_class'],
                        'query_minus_bida_under_mean':v['query_mean']-v['bida_mean'],
                        'class_minus_mean_under_query':v['query_class']-v['query_mean'],
                        'class_minus_mean_under_bida':v['bida_class']-v['bida_mean'],
                        'interaction':(v['query_class']-v['query_mean'])-(v['bida_class']-v['bida_mean'])}
    payload={'seed':2101,'contrasts':contrasts,'runs':runs,'probes':probes,'verified_artifacts':checked}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Tokenizer x readout factorial: stress seed2101','',
           'All cells use Full Joint BN and zero BiDA blocks. Single-seed diagnosis,',
           'not a three-seed stability estimate. query_class is reused; three cells newly trained.','',
           '## Fixed epoch200 primary','', '| Tokenizer | Readout | OA | AA | Kappa | Class6 share | Class7 recall | Recall coverage | Source-val OA |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,result in runs.items():
        row=result['target_at_final'];tok,head=arm.split('_')
        lines.append(f"| {tok} | {head} | {row['oa_percent']:.2f} | {row['aa_percent']:.2f} | {row['kappa']:.3f} | {100*row['prediction_shares'][5]:.2f}% | {row['per_class_percent'][6]:.2f}% | {sum(x>0 for x in row['per_class_percent'])}/7 | {result['source_val_final']['oa_percent']:.2f} |")
    lines+=['','## Factorial contrasts (percentage points)','', '| Contrast | OA | AA |','| --- | ---: | ---: |']
    for key in contrasts['oa_percent']:
        lines.append(f"| {key} | {contrasts['oa_percent'][key]:+.2f} | {contrasts['aa_percent'][key]:+.2f} |")
    lines+=['','Interaction = (Class minus Mean under Query) minus (Class minus Mean under BiDA).',
            'These are optimization-seed-specific descriptive differences, not confidence intervals.','',
            '## Oracle diagnostic only','', '| Arm | Best epoch | OA | Associated AA |','| --- | ---: | ---: | ---: |']
    for arm,r in runs.items():
        o=r['target_oracle'];m=o['metrics'];lines.append(f"| {arm} | {o['best_epoch']} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} |")
    lines+=['','## Fixed per-class recall','', '| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,r in runs.items():lines.append(f'| {arm} | '+' | '.join(f'{x:.2f}' for x in r['target_at_final']['per_class_percent'])+' |')
    lines+=['','## Fixed prediction distribution','', '| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,result in runs.items():
        r=result['target_at_final'];lines.append(f'| {arm} | '+' | '.join(f'{n} ({100*s:.2f}%)' for n,s in zip(r['prediction_counts'],r['prediction_shares']))+' |')
    lines+=['','## Fixed angular probes','', '| Arm | Mode | Domain | Token concentration | Pre-LN concentration | z norm | z concentration | Cos(z,w6) | Margin6 | Class6 share |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,p in probes.items():
        for mode,domains in p['rows'][-1]['modes'].items():
            for domain,r in domains.items():
                lines.append(f"| {arm} | {mode} | {domain} | {r['token_mean']['direction_concentration']:.3f} | {r['pre_norm']['direction_concentration']:.3f} | {r['z']['mean_norm']:.3f} | {r['z']['direction_concentration']:.3f} | {r['z']['cos_w6_mean']:.3f} | {r['margin6_mean']:.3f} | {100*r['prediction_shares'][5]:.2f}% |")
    lines+=['','## Trajectory collapse and parameters','', '| Arm | Class6 collapse grid points | Any-class ≥95% grid points | Parameters |','| --- | ---: | ---: | ---: |']
    for arm,r in runs.items():
        t=r['target_trajectory'];lines.append(f"| {arm} | {sum(x['metrics']['prediction_shares'][5]>=.95 for x in t)}/20 | {sum(max(x['metrics']['prediction_shares'])>=.95 for x in t)}/20 | {r['parameters']:,} |")
    lines+=['','## Interpretation of this stress-seed factorial','',
            'The same Class Query readout is compatible with balanced target coverage when fed BiDA tokens: AA73.12%, class6 share51.30%, class7 recall50.83%. With Query tokens the reused cell has AA47.24%, class6 share83.41%, class7 recall0%. This argues against Class Query being a tokenizer-independent sole cause.',
            'Replacing Class Query with mean pooling does not repair the Query cell: OA falls4.41pp and AA falls4.96pp, with class7 recall still0%. Query z direction concentration is0.885 under Mean versus0.867 under Class Query.',
            'The completed BiDA+Mean cell retains endpoint AA72.17% and seven-class recall, with z concentration0.510 versus0.504 under BiDA+Class Query. Both Query cells have much higher endpoint target concentration than both BiDA cells. The reader-independent adverse class-balanced endpoint signature therefore tracks the Query tokenizer package under this Full Joint setting.',
            'Do not label BiDA+Mean globally stable: at epoch170 target OA drops to14.03%, AA40.91%, with64.37% predictions in class4 and zero class6 predictions, before recovering. Its source CE spikes and source-val OA is70.87% at that checkpoint. A class6-specific collapse threshold misses this alternative transient failure.',
            'OA alone gives a misleading tokenizer ranking under mean pooling: Query exceeds BiDA by5.68pp OA but loses29.89pp AA. BiDA+Mean class6 recall is58.87%, versus80.24% with Class Query; the dominant target class heavily influences OA. Class Query improves BiDA OA by11.35pp and AA by0.95pp rather than causing collapse.',
            'Current joint-training moments reproduce the high Query concentration/cosine signatures, so saved inference buffers alone do not explain this pattern. Stage probes are observational and use the same unlabeled sample in all cells.',
            'Provisional diagnosis: prioritize the spatial Query package and its learned domain representation, not removal of Class Query or extra losses. This does not isolate learnable queries from positional encoding, attention, LayerNorm, FFN or parameter count, and does not prove Query is intrinsically flawed. All learned components co-adapt; one stress seed is insufficient for a general attribution.',
            '', '## Scope and checks','',
            f'Verified {checked} checkpoint/prediction SHA256 records and four manifests. All new predictions frozen before post-hoc GT metrics.',
            'Common pipeline initialization matches the reused control; cell selection preserves RNG state. Retained stem/classifier/head tensors exactly match at initialization.',
            'BiDA/Query is a tokenizer package comparison; Class Query/Mean changes capacity and nonlinear readout. All trainable state co-adapts. One stress seed selected after prior outcomes cannot establish general causality or stability.',
            'Fixed samples and source/target probes are shared. Eval uses saved buffers; joint-train probe uses current moments with dropout disabled and restores/hash-checks state. Not live historical activations.',
            'No added loss, BN change, residual scale, ratio or oracle tuning. The main reference remains Full Joint BN + BiDA tokenizer with its original depth3: 79.50±1.46% over three seeds.']
    Path(__file__).with_name('RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for arm,r in runs.items():
        t=r['target_trajectory'];p=probes[arm]['rows']
        for axis,key in zip(axes[0],('oa_percent','aa_percent','prediction_shares')):
            v=[100*x['metrics'][key][5] if key=='prediction_shares' else x['metrics'][key] for x in t]
            axis.plot([x['epoch'] for x in t],v,label=arm);axis.set_title(key);axis.set_ylim(0,103);axis.grid(alpha=.2)
        for axis,key in zip(axes[1],('direction_concentration','cos_w6_mean','margin6_mean')):
            v=[x['modes']['eval']['target']['z'][key] if key!='margin6_mean' else x['modes']['eval']['target'][key] for x in p]
            axis.plot([x['epoch'] for x in p],v,label=arm);axis.set_title('Target '+key);axis.grid(alpha=.2);axis.set_xlabel('Epoch')
    axes[0,2].axhline(95,color='red',linestyle='--',linewidth=.8);axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'trajectories.png',dpi=160);plt.close(fig)
    print(json.dumps({'fixed':{a:{k:r['target_at_final'][k] for k in ('oa_percent','aa_percent','kappa')} for a,r in runs.items()},'contrasts':contrasts},indent=2))

if __name__=='__main__':main()
