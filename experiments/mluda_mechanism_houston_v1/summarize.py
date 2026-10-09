"""Post-hoc scoring only once every final prediction is frozen and audits pass."""
import json

import numpy as np
from sklearn import metrics

from common import OUT, SEEDS, ARMS, dump, sha


def main():
    records = {}
    for seed in SEEDS:
        paired = []
        for arm in ARMS:
            folder=OUT/('%s_%d'%(arm,seed))
            frozen=json.loads((folder/'frozen.json').read_text())
            for name,key in [('predictions.npy','predictions_sha256'),('labels.npy','labels_sha256'),('epoch100.pth','checkpoint_sha256')]:
                assert sha(folder/name)==frozen[key]
            init=json.loads((folder/'initialization.json').read_text())
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
            assert len(history)==100 and history[-1]['updates']==3800
            paired.append((frozen,init,history))
            records[(arm,seed)]={'frozen':frozen,'folder':folder}
        base=paired[0]
        for f,i,h in paired[1:]:
            for key in ('samples','test_source_pair_sha256','batch_augmentation_rng_sha256','labels_sha256','native_target_dataset_n','scored_n'):
                assert f[key]==base[0][key],(seed,key)
            for key in ('native_state_sha256','native_rng_sha256'):
                assert i[key]==base[1][key],(seed,key)
            for a,b in zip(h,base[2]):
                assert a['epoch']==b['epoch']
                for key in ('batch_augmentation_rng_sha256','end_rng_sha256'):
                    assert a[key]==b[key],(seed,a['epoch'],key)
    dump(OUT/'paired_audit.json',{'passed':True,'all_12_frozen_before_scoring':True,
        'same_seed_native_initialization_samples_batches_augmentation_rng_test_source_pair':True})
    rows=[]
    for (arm,seed),r in records.items():
        f=r['frozen']; folder=r['folder']
        y=np.load(folder/'labels.npy'); p=np.load(folder/'predictions.npy')
        c=metrics.confusion_matrix(y,p,labels=list(range(7)))
        recall=100*np.diag(c)/c.sum(1)
        shares=np.bincount(p,minlength=7)/len(p)
        row={'arm':arm,'seed':seed,'oa':100*float((y==p).sum())/f['native_target_dataset_n'],
             'aa':float(recall.mean()),'kappa':float(metrics.cohen_kappa_score(y,p)),
             'recall':recall.tolist(),'prediction_shares':shares.tolist(),'confusion':c.tolist(),
             'collapse':bool(shares.max()>=.95),'zero_recall_classes':(np.where(recall==0)[0]+1).tolist(),
             'parameters':f['parameters'],'training_seconds':f['training_seconds'],
             'evaluation_seconds':f['evaluation_seconds'],'evaluated_n':len(p),'native_denominator_n':f['native_target_dataset_n']}
        rows.append(row); dump(folder/'metrics.json',row)
    index={(r['arm'],r['seed']):r for r in rows}
    summary={}
    for arm in ARMS:
        rs=[index[(arm,s)] for s in SEEDS]
        summary[arm]={k:{'mean':float(np.mean([r[k] for r in rs])),
              'population_std':float(np.std([r[k] for r in rs],ddof=0)),
              'sample_std':float(np.std([r[k] for r in rs],ddof=1))}
              for k in ('oa','aa','kappa','training_seconds','evaluation_seconds')}
        summary[arm]['collapse_count']=sum(r['collapse'] for r in rs)
    deltas={}
    for plus,minus in [('M1','M0'),('M2','M0'),('M3','M1'),('M3','M2')]:
        ds=[{'seed':s,**{k:index[(plus,s)][k]-index[(minus,s)][k] for k in ('oa','aa','kappa')},
             'recall_delta':(np.array(index[(plus,s)]['recall'])-np.array(index[(minus,s)]['recall'])).tolist()} for s in SEEDS]
        deltas[plus+'-'+minus]={'per_seed':ds,'summary':{k:{'mean':float(np.mean([d[k] for d in ds])),
                     'sample_std':float(np.std([d[k] for d in ds],ddof=1))} for k in ('oa','aa','kappa')}}
    comparison=[{'historical_reuse':False,'reason':'No strictly paired Houston artifact bundle; all three M0 rebuilt.'}]
    result={'runs':rows,'summary':summary,'paired_deltas':deltas,'historical_M0_comparison':comparison}
    dump(OUT/'summary.json',result)
    def fmt(v,precision=3):
        return ('%.*f ± %.*f'%(precision,v['mean'],precision,v['sample_std']))
    lines=['# Full MLUDA + mechanisms, Houston13 → Houston18','',
           'Fixed epoch100, seeds1174/1370/1417; means ± sample SD. All12 final predictions frozen before scoring.', '',
           '| Arm | OA (%) | AA (%) | Kappa | Collapse | Parameters | Train seconds |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        z=summary[arm]
        lines.append('| %s | %s | %s | %s | %d/3 | %d | %s |'%(arm,fmt(z['oa']),fmt(z['aa']),fmt(z['kappa'],5),z['collapse_count'],index[(arm,SEEDS[0])]['parameters'],fmt(z['training_seconds'],1)))
    lines+=['','| Arm/seed | OA | AA | Kappa | Zero-recall classes | Recall C1–C7 (%) | Prediction shares C1–C7 |',
            '| --- | ---: | ---: | ---: | --- | --- | --- |']
    for r in rows:
        lines.append('| %s/%d | %.3f | %.3f | %.5f | %s | %s | %s |'%(r['arm'],r['seed'],r['oa'],r['aa'],r['kappa'],r['zero_recall_classes'],', '.join('%.2f'%x for x in r['recall']),', '.join('%.4f'%x for x in r['prediction_shares'])))
    lines+=['','| Paired contrast/seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |','| --- | ---: | ---: | ---: |']
    for name,d in deltas.items():
        for r in d['per_seed']:
            lines.append('| %s/%d | %+.3f | %+.3f | %+.5f |'%(name,r['seed'],r['oa'],r['aa'],r['kappa']))
        z=d['summary']; lines.append('| %s mean ± SD | %s | %s | %s |'%(name,fmt(z['oa']),fmt(z['aa']),fmt(z['kappa'],5)))
    lines+=['','Native OA retains full dataset denominator despite drop_last; AA/Kappa use retained predictions.',
            'Joint BN .19 matches old-statistic decay of two .1 calls; moments/normalization are NOT equivalent.',
            'Original ILDA, augmentation (including native axis behavior), full losses and optimizer recreation retained.',
            'Historical M0 comparison (no historical checkpoint/stream reuse):','',json.dumps(comparison,indent=2),'',
            'All endpoints/negative results retained. No target-based tuning/selection/additional experiments.',
            'Timings include instrumentation and concurrent GPU load; not an isolated throughput benchmark.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
