"""All arms retained; fixed epoch only, no target best selection."""
import csv
import json
import numpy as np
import common as c


def main():
    runs={}; rows=[]
    for arm in c.ARMS:
        for seed in (2100,2101,2102):
            run=c.OUT/('%s_%d'%(arm,seed))
            assert (run/'results.json').exists(), 'Wait for all twelve tasks'
            runs[(arm,seed)]=json.loads((run/'results.json').read_text())
            for row in json.loads((run/'trajectory.json').read_text()):
                rows.append({'arm':arm,'seed':seed,'epoch':row['epoch'],
                    'oa':row['oa_percent'],'aa':row['aa_percent'],'kappa':row['kappa'],
                    **{'recall_%d'%(i+1):v for i,v in enumerate(row['per_class_percent'])},
                    **{'share_%d'%(i+1):v for i,v in enumerate(row['prediction_shares'])},
                    **{'cm_%d_%d'%(i+1,j+1):v for i,vs in enumerate(row['confusion_matrix']) for j,v in enumerate(vs)},
                    **{k:row[k] for k in ('error_3_to_7_count','error_3_to_7_ratio','error_4_to_7_count','error_4_to_7_ratio','single_class_collapse','zero_recall_class_count')}})
    with (c.OUT/'trajectory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries={}; paired=[]; hashes=[]
    lines=['# Pavia normalization × source sampling: fixed epoch 200','',
           'Twelve frozen tasks. Sample standard deviation (ddof=1). No target-best selection.','',
           '| Arm | OA | AA | Kappa | Single-class collapse |','|---|---:|---:|---:|---:|']
    for arm in c.ARMS:
        values=[runs[(arm,s)]['target_at_final'] for s in (2100,2101,2102)]
        summaries[arm]={k:{'mean':float(np.mean([v[k] for v in values])),
                           'sample_std':float(np.std([v[k] for v in values],ddof=1))}
                        for k in ('oa_percent','aa_percent','kappa')}
        cells=['%.3f ± %.3f'%(summaries[arm][k]['mean'],summaries[arm][k]['sample_std']) for k in summaries[arm]]
        lines.append('| %s | %s | %d/3 |'%(arm,' | '.join(cells),sum(v['single_class_collapse'] for v in values)))
    for seed in (2100,2101,2102):
        histories={a:[json.loads(x) for x in (c.OUT/('%s_%d'%(a,seed))/'history.jsonl').read_text().splitlines()] for a in c.ARMS}
        assert all(len(h)==200 and all(x['steps']==290 for x in h) for h in histories.values())
        for epoch in range(200):
            for key in ('target_center_sha256','target_augmentation_sha256','model_rng_sha256'):
                assert len({histories[a][epoch][key] for a in c.ARMS})==1,(seed,epoch,key)
            for a,b in [('P0','P1'),('P2','P3')]:
                for key in ('source_center_sha256','source_augmentation_sha256'):
                    assert histories[a][epoch][key]==histories[b][epoch][key]
        hashes.append({'seed':seed,'all_200_epoch_sequence_and_model_rng_checks_passed':True})
        for a,b in [('P1','P0'),('P2','P0'),('P3','P2'),('P3','P1')]:
            va,vb=runs[(a,seed)]['target_at_final'],runs[(b,seed)]['target_at_final']
            paired.append({'seed':seed,'contrast':a+'-'+b,
                **{k:va[k]-vb[k] for k in ('oa_percent','aa_percent','kappa','error_3_to_7_ratio','error_4_to_7_ratio')},
                'recall_delta':[x-y for x,y in zip(va['per_class_percent'],vb['per_class_percent'])]})
    c.dump(c.OUT/'summary.json',{'summary':summaries,'paired_deltas':paired,'stream_checks':hashes,
        'all_runs':{'%s_%d'%k:v for k,v in runs.items()}})
    lines+=['','Full recalls, confusion matrices, error ratios, coverage, times and paired contrasts: summary.json / trajectory.csv.',
            'All 200-epoch target/augmentation/model RNG and paired source-center checks passed.',
            'No configuration is automatically selected or additional experiment scheduled.']
    (c.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
