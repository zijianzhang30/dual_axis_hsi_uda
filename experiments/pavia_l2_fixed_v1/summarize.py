"""Report all nine fixed-200 runs, retaining negative and positive results alike."""
import json
import numpy as np
from scipy.io import loadmat
import common as c


def stats(values):
    return {'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1))}


def main():
    assert not (c.OUT/'summary.json').exists()
    configs={};manifests={};histories={}
    for seed in (2100,2101,2102):
        for arm in ('A','B','C'):
            name='%s_%d'%(arm,seed);folder=c.OUT/name
            cfg=json.loads((folder/'config.json').read_text())
            manifest=json.loads((folder/'frozen_manifest.json').read_text())
            assert cfg['seed']==seed and cfg['epochs']==200 and cfg['arm']==arm
            assert cfg['normalization']=='native spectral L2; repaired flat-to-cube return'
            assert cfg['primary']=='fixed_epoch_200' and cfg['loss']=='source CE only'
            assert manifest['config']==cfg
            assert manifest['prediction']['sha256']==c.helper.file_hash(folder/'predictions_epoch_200.npz')
            for record in manifest['checkpoints']:
                assert record['sha256']==c.helper.file_hash(folder/'checkpoints'/('epoch_%03d.pth'%record['epoch']))
            hist=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            assert len(hist)==200 and [x['epoch'] for x in hist]==list(range(1,201))
            assert all(x['steps']==290 for x in hist)
            configs[name],manifests[name],histories[name]=cfg,manifest,hist
        for arm in ('B','C'):
            for field in ('initial_model_sha256','data_sha256','indices_sha256','sample_counts',
                          'normalization','lr','batch_size','depth','dropout','common_sha256'):
                assert configs['A_%d'%seed][field]==configs['%s_%d'%(arm,seed)][field],field
            for a,b in zip(histories['A_%d'%seed],histories['%s_%d'%(arm,seed)]):
                for field in ('epoch','steps','batch_sha256','rng_sha256'):
                    assert a[field]==b[field],(seed,arm,a['epoch'],field)
    # Independently reconstruct native strict boundary mask; do not iterate loaders.
    gt=loadmat(c.DATA/'Pavia/pavia_gt_7.mat')['pavia_gt_7'].astype(int)-1
    indices=np.argwhere(gt>=0);indices=indices[(indices[:,0]>0)&(indices[:,1]>0)]
    labels=gt[indices[:,0],indices[:,1]]
    assert len(labels)==39348
    rows={}
    for name,cfg in configs.items():
        folder=c.OUT/name
        with np.load(folder/'predictions_epoch_200.npz') as prediction:
            metrics=c.helper.metrics(labels,prediction['predictions'],prediction['logits'],7)
        previous=json.loads((folder/'results.json').read_text())
        for field in ('oa_percent','aa_percent','kappa','per_class_percent','prediction_shares'):
            assert np.allclose(metrics[field],previous['target_at_final'][field],atol=1e-10,rtol=0)
        metrics.update(single_class_collapse=max(metrics['prediction_shares'])>=.95,
                       zero_recall_classes=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0])
        rows[name]={'metrics':metrics,'source_val':previous['source_val_at_final'],
                    'parameters':cfg['trainable_parameters'],'train_seconds':manifests[name]['train_seconds'],
                    'prediction_sha256':manifests[name]['prediction']['sha256'],
                    'history_sha256':c.helper.file_hash(folder/'history.jsonl')}
    fields=('oa_percent','aa_percent','kappa')
    summaries={arm:{f:stats([rows['%s_%d'%(arm,s)]['metrics'][f] for s in (2100,2101,2102)]) for f in fields}
               for arm in ('A','B','C')}
    paired={}
    for name,left,right in (('B_minus_A','B','A'),('C_minus_B','C','B'),('C_minus_A','C','A')):
        deltas={f:[rows['%s_%d'%(left,s)]['metrics'][f]-rows['%s_%d'%(right,s)]['metrics'][f]
                   for s in (2100,2101,2102)] for f in fields}
        paired[name]={'by_seed':deltas,'summary':{f:stats(v) for f,v in deltas.items()}}
    result={'benchmark':'PaviaU -> PaviaC, repaired L2 BiDA-style protocol','epoch':200,
            'runs':rows,'summary':summaries,'paired':paired,
            'all200_epoch_augmented_batch_and_RNG_pairing_passed':True,
            'historical_raw_input_results_not_reused':True}
    c.dump(c.OUT/'summary.json',result)
    lines=['# Repaired Pavia L2 validation: fixed epoch 200','',
           'All nine runs retained; mean +/- sample standard deviation. No target-selected checkpoints.',
           'This is BiDA-style validation, not native MLUDA protocol reproduction.', '',
           '| Arm | OA (%) | AA (%) | Kappa | Single-class collapse |',
           '| --- | ---: | ---: | ---: | ---: |']
    for arm in ('A','B','C'):
        values=['%.3f +/- %.3f'%(summaries[arm][f]['mean'],summaries[arm][f]['sample_std']) for f in fields]
        collapse=sum(rows['%s_%d'%(arm,s)]['metrics']['single_class_collapse'] for s in (2100,2101,2102))
        lines.append('| %s | %s | %s | %s | %d/3 |'%(arm,*values,collapse))
    lines+=['','A: Original Mixed BN + original tokenizer. B: Full Joint BN + original tokenizer.',
            'C: Full Joint BN + center-relative tokenizer.', '',
            '| Seed | B-A OA / AA (pp) | C-B OA / AA (pp) | C-A OA / AA (pp) |',
            '| --- | ---: | ---: | ---: |']
    for i,seed in enumerate((2100,2101,2102)):
        values=['%+.3f / %+.3f'%(paired[name]['by_seed']['oa_percent'][i],paired[name]['by_seed']['aa_percent'][i])
                for name in ('B_minus_A','C_minus_B','C_minus_A')]
        lines.append('| %d | %s | %s | %s |'%(seed,*values))
    for name,row in rows.items():
        m=row['metrics']
        lines+=['','## '+name,'',
                'OA %.3f; AA %.3f; Kappa %.5f; source-val OA %.3f; parameters %d; train seconds %.1f.'%
                (m['oa_percent'],m['aa_percent'],m['kappa'],row['source_val']['oa_percent'],row['parameters'],row['train_seconds']),
                'Recall (classes 1-7): '+', '.join('%.2f'%v for v in m['per_class_percent']),
                'Prediction shares (%): '+', '.join('%.2f'%(v*100) for v in m['prediction_shares']),
                'Zero-recall classes: '+str(m['zero_recall_classes'])]
    lines+=['','Paired summaries:','',json.dumps(paired,indent=2),
            '', 'Timing includes concurrent execution; not an isolated speed benchmark.',
            'No automatic further training or version choice.']
    (c.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'summary':summaries,'paired':paired},indent=2))


if __name__=='__main__':main()
