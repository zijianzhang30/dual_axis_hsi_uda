"""Score all frozen fixed-200 predictions, audit pairing, report every arm."""
import argparse
import json
from pathlib import Path
import numpy as np
import train

ROOT=train.ROOT
OUT=ROOT/'results/relation_tokenizer_v1'
SEEDS=(2100,2101,2102)


def stats(values):
    return {'values':list(values),'mean':float(np.mean(values)),
            'sample_sd':float(np.std(values,ddof=1))}


def fmt(values,digits=2):
    r=stats(values)
    return f"{r['mean']:.{digits}f} ± {r['sample_sd']:.{digits}f}"


def main():
    runs={}
    for arm in ('A','B','C'):
        for seed in SEEDS:
            folder=OUT/f'{arm}_{seed}'
            m=json.loads((folder/'frozen_manifest.json').read_text())
            c=m['config']
            h=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            assert len(h)==200 and all(x['steps']==18 for x in h)
            assert [x['epoch'] for x in h]==list(range(1,201))
            assert [x['epoch'] for x in m['checkpoints']]==list(range(10,201,10))
            for item in [*m['checkpoints'],m['prediction']]:
                assert train.helper.file_hash(item['path'])==item['sha256']
            assert c['parameters']==(376567 if arm=='A' else 378011)
            if arm=='A':assert m['historical_replay']['historical_state_exact']
            runs[arm,seed]={'manifest':m,'history':h}
    for seed in SEEDS:
        for arm in ('B','C'):
            a=runs['A',seed];b=runs[arm,seed]
            for key in ('initial_original_model_sha256','source_n','source_val_n','target_n',
                        'code_sha256','data_sha256','depth','dropout','lr','bn_momentum'):
                assert a['manifest']['config'][key]==b['manifest']['config'][key]
            for x,y in zip(a['history'],b['history']):
                for key in ('batch_stream_sha256','original_rng_stream_sha256'):
                    assert x[key]==y[key],(seed,arm,x['epoch'],key)
        assert runs['B',seed]['manifest']['config']['initial_relation_mlp_sha256']==runs['C',seed]['manifest']['config']['initial_relation_mlp_sha256']
    # This combined manifest is written before any target labels are collected.
    frozen={'primary':'fixed_epoch_200','runs':{f'{a}_{s}':{'manifest_sha256':train.helper.file_hash(OUT/f'{a}_{s}/frozen_manifest.json'),
            'prediction_sha256':r['manifest']['prediction']['sha256']} for (a,s),r in runs.items()}}
    combined=OUT/'all_predictions_frozen.json'
    combined.write_text(json.dumps(frozen,indent=2)+'\n')
    args=argparse.Namespace(seed=2100,num_workers=4,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
    train.seed_worker(2100)
    _,_,_,loader,_,classes=train.make_loaders(args)
    labels=np.concatenate([y.numpy() for _,y in loader])
    for (arm,seed),r in runs.items():
        with np.load(r['manifest']['prediction']['path']) as artifact:
            metrics=train.helper.metrics(labels,artifact['predictions'],artifact['logits'],classes)
        metrics['class6_collapse']=metrics['prediction_shares'][5]>=.95
        metrics['any_class_collapse']=max(metrics['prediction_shares'])>=.95
        metrics['zero_recall_classes']=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0]
        m=r['manifest']
        result={'target_at_final':metrics,'source_val_at_final':m['source_val_at_final'],
                'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],
                'inference_seconds':m['inference_seconds'],'frozen_manifest_sha256':frozen['runs'][f'{arm}_{seed}']['manifest_sha256'],
                'all_predictions_manifest_sha256':train.helper.file_hash(combined)}
        (OUT/f'{arm}_{seed}/results.json').write_text(json.dumps(result,indent=2)+'\n')
        r['result']=result
    keys=('oa_percent','aa_percent','kappa')
    aggregate={}
    for arm in ('A','B','C'):
        rs=[runs[arm,s]['result'] for s in SEEDS]
        aggregate[arm]={k:stats([r['target_at_final'][k] for r in rs]) for k in keys}
        aggregate[arm].update({'class6_collapse_count':sum(r['target_at_final']['class6_collapse'] for r in rs),
          'any_class_collapse_count':sum(r['target_at_final']['any_class_collapse'] for r in rs),
          'parameters':rs[0]['parameters'],'train_seconds':stats([r['train_seconds'] for r in rs]),
          'source_val_oa':stats([r['source_val_at_final']['oa_percent'] for r in rs])})
    deltas={name:{k:stats([runs['C',s]['result']['target_at_final'][k]-runs[right,s]['result']['target_at_final'][k] for s in SEEDS])
                  for k in keys} for name,right in (('C-A','A'),('C-B','B'))}
    summary={'aggregate':aggregate,'paired_deltas':deltas,'runs':{f'{a}_{s}':r['result'] for (a,s),r in runs.items()},
             'all_200_epoch_pairing_audit':'passed','all_A_historical_state_replays':'exact',
             'target_scoring':'only after all nine predictions frozen','selection':'none'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Relation tokenizer V1: Houston13 → Houston18', '',
           'Fixed epoch 200, seeds 2100/2101/2102. Mean ± sample std (ddof=1).',
           'A: original weights; B: local-standardized relation weights; C: center-relative relation weights.',
           'All use unchanged CNN values, Full Joint BN, depth3 self Transformer and source CE only.', '',
           '| Arm | OA (%) | AA (%) | Kappa | Class-6 collapse | Any-class collapse |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('A','B','C'):
        rs=[runs[arm,s]['result']['target_at_final'] for s in SEEDS]
        lines.append(f'| {arm} | '+' | '.join(fmt([r[k] for r in rs],3 if k=='kappa' else 2) for k in keys)+f" | {aggregate[arm]['class6_collapse_count']}/3 | {aggregate[arm]['any_class_collapse_count']}/3 |")
    lines+=['','## Individual endpoints','','| Arm | Seed | OA | AA | Kappa | Class-6 share (%) | Zero-recall classes | Source-val OA |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for (arm,seed),r in runs.items():
        m=r['result']['target_at_final']
        lines.append(f"| {arm} | {seed} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} | {m['kappa']:.3f} | {100*m['prediction_shares'][5]:.2f} | {m['zero_recall_classes'] or 'none'} | {r['result']['source_val_at_final']['oa_percent']:.2f} |")
    for name,right in (('C-A','A'),('C-B','B')):
        lines+=['',f'## Paired {name}','','| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |','| --- | ---: | ---: | ---: |']
        for i,seed in enumerate(SEEDS):
            lines.append(f'| {seed} | '+' | '.join(f"{deltas[name][k]['values'][i]:+.3f}" for k in keys)+' |')
        lines.append('| Mean ± std | '+' | '.join(fmt(deltas[name][k]['values'],3 if k=='kappa' else 2) for k in keys)+' |')
    for arm in ('A','B','C'):
        for label,key,factor in (('Recall','per_class_percent',1),('Prediction share','prediction_shares',100)):
            lines+=['',f'## {arm}: {label} (%)','','| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |',
                    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for seed in SEEDS:
                lines.append(f'| {seed} | '+' | '.join(f'{factor*v:.2f}' for v in runs[arm,seed]['result']['target_at_final'][key])+' |')
    lines+=['','## Parameters, cost and audit','','| Arm | Parameters | Training seconds | Source-val OA |',
            '| --- | ---: | ---: | ---: |']
    for arm in ('A','B','C'):
        r=aggregate[arm]
        lines.append(f"| {arm} | {r['parameters']} | {fmt(r['train_seconds']['values'])} | {fmt(r['source_val_oa']['values'])} |")
    lines+=['','Concurrent GPU wall times are observational, not an isolated speed benchmark.',
            'B/C replace the 256-parameter conv_a with a 1700-parameter pointwise MLP; +1444 parameters.',
            'Implementation numerical, spatial alignment, finite-gradient and target-joint-statistics gradient checks passed.',
            'A freshly replayed all three historical final models exactly, including BN buffers.',
            'All three arms match same-seed batch and pre-forward RNG hashes for all 200 epochs.',
            'All 180 checkpoint and nine prediction hashes verified before target scoring.',
            'No Pavia files or processes changed; no target-based tuning, early stopping or version selection.',
            'No new experiments are automatically appended from these results.',
            'Full configs/logs/manifests/code snapshots/diff and summary: `results/relation_tokenizer_v1/`.']
    print('\n'.join(lines))


if __name__=='__main__':main()
