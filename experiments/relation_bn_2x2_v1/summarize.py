"""Audit all twelve trajectories, then score fixed200 and oracle separately."""
import argparse
import json
import numpy as np
import train
import audit
from grid import cell_folder

OUT=audit.OUT
SEEDS=audit.SEEDS
CELLS=('1','2','3','4')
NAMES={'1':'Original BN + original tokenizer','2':'Original BN + center-relative',
       '3':'Full Joint BN + original tokenizer','4':'Full Joint BN + center-relative'}


def stats(values):return {'values':list(values),'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}


def fmt(values,digits=2):
    r=stats(values);return f"{r['mean']:.{digits}f} ± {r['sample_sd']:.{digits}f}"


def main():
    audit_result=audit.run() # Recheck that reused historical artifacts remain unchanged.
    runs={};records=[]
    for seed in SEEDS:
        old_folder=audit.folder('4',seed)
        old=audit.read(old_folder/'config.json')
        new_folder=cell_folder('2',seed);new=audit.read(new_folder/'config.json')
        for key in ('seed','epochs','initial_original_model_sha256','initial_full_model_sha256',
                    'initial_relation_mlp_sha256','parameters','source_n','source_val_n','target_n',
                    'eps','relation_hidden','tokens','dim','depth','dropout','lr','loss','primary','data_sha256'):
            assert old[key]==new[key],(seed,key)
        histories={c:[json.loads(s) for s in (cell_folder(c,seed)/'history.jsonl').read_text().splitlines()] for c in CELLS}
        assert all(len(h)==200 for h in histories.values())
        for i in range(200):
            for key in ('batch_stream_sha256','original_rng_stream_sha256'):
                assert len({histories[c][i][key] for c in CELLS})==1
        for c in CELLS:
            manifest=audit.read(cell_folder(c,seed)/('training_manifest.json' if c=='2' else 'frozen_manifest.json'))
            old_result=audit.read(cell_folder(c,seed)/'results.json') if c!='2' else None
            runs[c,seed]={'config':audit.read(cell_folder(c,seed)/'config.json'),
                'source_val_at_final':histories[c][-1]['source_val'],
                'train_seconds':manifest['train_seconds'] if c!='1' else old_result['train_seconds'],
                'parameters':378011 if c in ('2','4') else 376567,'trajectory':[]}
    for cell in CELLS:
        m=audit.read(OUT/f'grid_cell{cell}/frozen_manifest.json')
        assert len(m['records'])==60
        for r in m['records']:
            assert audit.sha(r['checkpoint_path'])==r['checkpoint_sha256']
            assert audit.sha(r['prediction_path'])==r['prediction_sha256']
        records+=m['records']
    # Commit hashes of all 240 outputs before reading any target values for evaluation.
    combined=OUT/'all_grid_predictions_frozen.json'
    combined.write_text(json.dumps({'records':records,'reuse_audit':audit_result},indent=2)+'\n')
    args=argparse.Namespace(seed=2100,num_workers=4,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
    train.base.seed_worker(2100)
    _,_,_,loader,_,classes=train.base.make_loaders(args)
    labels=np.concatenate([y.numpy() for _,y in loader])
    for r in records:
        with np.load(r['prediction_path']) as a:
            metrics=train.base.helper.metrics(labels,a['predictions'],a['logits'],classes)
        metrics['class6_collapse']=metrics['prediction_shares'][5]>=.95
        metrics['any_class_collapse']=max(metrics['prediction_shares'])>=.95
        metrics['zero_recall_classes']=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0]
        runs[r['cell'],r['seed']]['trajectory'].append({'epoch':r['epoch'],'metrics':metrics})
    for (c,s),r in runs.items():
        r['trajectory'].sort(key=lambda x:x['epoch'])
        assert [x['epoch'] for x in r['trajectory']]==list(range(10,201,10))
        r['target_at_final']=r['trajectory'][-1]['metrics']
        r['target_oracle']=max(r['trajectory'],key=lambda x:x['metrics']['oa_percent'])
        if c!='2':
            historical=audit.read(cell_folder(c,s)/'results.json')['target_at_final']
            for key in ('oa_percent','aa_percent','kappa','per_class_percent','prediction_shares'):
                assert r['target_at_final'][key]==historical[key],(c,s,key)
        (OUT/f'cell{c}_{s}_evaluation.json').write_text(json.dumps(r,indent=2)+'\n')
    keys=('oa_percent','aa_percent','kappa')
    aggregate={c:{k:stats([runs[c,s]['target_at_final'][k] for s in SEEDS]) for k in keys} for c in CELLS}
    deltas={name:{k:stats([runs[left,s]['target_at_final'][k]-runs[right,s]['target_at_final'][k] for s in SEEDS]) for k in keys}
            for name,left,right in (('2-1','2','1'),('4-2','4','2'))}
    result={'primary':'fixed_epoch_200','secondary':'target-oracle OA on grid10:10:200; diagnostic only',
            'aggregate':aggregate,'paired_deltas':deltas,'runs':{f'cell{c}_{s}':r for (c,s),r in runs.items()},
            'audit':'all historical/current initialization and 200-epoch batch/RNG checks passed',
            'prediction_grid_hashes_verified':240,'new_training_tasks':3,'automatic_version_selection':False}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Tokenizer × BN 2×2: Houston13 → Houston18','','Fixed epoch 200 is primary. Mean ± sample std(ddof=1), seeds2100/2101/2102.',
           'Only cell2 adds training; cells1/3/4 are audited historical reuse. No new modules or losses.', '',
           '| Cell | OA (%) | AA (%) | Kappa | Class-6 collapse | Any-class collapse |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in CELLS:
        ms=[runs[c,s]['target_at_final'] for s in SEEDS]
        lines.append(f'| {c}: {NAMES[c]} | '+' | '.join(fmt([m[k] for m in ms],3 if k=='kappa' else 2) for k in keys)+f" | {sum(m['class6_collapse'] for m in ms)}/3 | {sum(m['any_class_collapse'] for m in ms)}/3 |")
    lines+=['','## Per-seed fixed200','','| Cell | Seed | OA | AA | Kappa | Class-6 share (%) | Zero-recall classes | Source-val OA |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for c in CELLS:
        for s in SEEDS:
            r=runs[c,s];m=r['target_at_final']
            lines.append(f"| {c} | {s} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} | {m['kappa']:.3f} | {100*m['prediction_shares'][5]:.2f} | {m['zero_recall_classes'] or 'none'} | {r['source_val_at_final']['oa_percent']:.2f} |")
    for name in ('2-1','4-2'):
        lines+=['',f'## Paired {name}','','| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |','| --- | ---: | ---: | ---: |']
        for i,s in enumerate(SEEDS):lines.append(f'| {s} | '+' | '.join(f"{deltas[name][k]['values'][i]:+.3f}" for k in keys)+' |')
        lines.append('| Mean ± std | '+' | '.join(fmt(deltas[name][k]['values'],3 if k=='kappa' else 2) for k in keys)+' |')
    lines+=['','## Target-oracle diagnostic, NOT primary or target-blind selection','',
            'Max target OA among epochs10,20,...,200, earliest epoch on ties; AA/Kappa are from the same checkpoint.',
            'Official local BiDA code additionally evaluates epoch1; that epoch is not part of this agreed grid.', '',
            '| Cell | Seed | Oracle epoch | Oracle OA | Associated AA | Associated Kappa |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for c in CELLS:
        for s in SEEDS:
            o=runs[c,s]['target_oracle'];m=o['metrics']
            lines.append(f"| {c} | {s} | {o['epoch']} | {m['oa_percent']:.2f} | {m['aa_percent']:.2f} | {m['kappa']:.3f} |")
        lines.append(f"| {c} mean ± std | — | — | {fmt([runs[c,s]['target_oracle']['metrics']['oa_percent'] for s in SEEDS])} | {fmt([runs[c,s]['target_oracle']['metrics']['aa_percent'] for s in SEEDS])} | {fmt([runs[c,s]['target_oracle']['metrics']['kappa'] for s in SEEDS],3)} |")
    for c in CELLS:
        for name,key,factor in (('Recall','per_class_percent',1),('Prediction share','prediction_shares',100)):
            lines+=['',f'## Cell{c} fixed200 {name} (%)','','| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |',
                    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for s in SEEDS:lines.append(f'| {s} | '+' | '.join(f'{factor*v:.2f}' for v in runs[c,s]['target_at_final'][key])+' |')
    lines+=['','## Costs and reproducibility','','| Cell | Parameters | Training seconds |','| --- | ---: | ---: |']
    for c in CELLS:lines.append(f"| {c} | {runs[c,2100]['parameters']} | {fmt([runs[c,s]['train_seconds'] for s in SEEDS])} |")
    lines+=['','Historical/new timings use different concurrent loads, not a controlled speed comparison.',
            'Native original BiDA source→target forward and BatchNorm counters verified; no replacement BN for cell2.',
            'All 200 epochs match same-seed batch/RNG streams across four cells; center-MLP initialization identical in cells2/4.',
            'All 240 grid predictions/checkpoints hashed and frozen before target scoring. Reused fixed200 metrics reproduce exactly.',
            'No target-based tuning, early stopping, formal checkpoint selection or automatic version promotion.',
            'Pavia and other historical artifacts were not modified. No additional training automatically appended.',
            'Inference encountered root-disk exhaustion near completion. Only this new result directory was moved to NAS, with a project symlink retained. Completed predictions were preserved; invalid partial writes were quarantined and missing predictions resumed without retraining.',
            f'Physical result directory: `{OUT.resolve()}`.',
            'Configs/logs/code snapshots/reuse audit/full trajectories/paired deltas: `results/relation_bn_2x2_v1/summary.json`.']
    print('\n'.join(lines))


if __name__=='__main__':main()
