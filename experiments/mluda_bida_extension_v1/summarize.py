"""Summarize all prescribed runs; never rank/select or trigger more training."""
import argparse
import json
import numpy as np
import common as c


def stats(rows):
    return {k:{'mean':float(np.mean([r[k] for r in rows])),
               'sample_std':float(np.std([r[k] for r in rows],ddof=1))}
            for k in ('oa_percent','aa_percent','kappa')}


def profile(metric):
    zeros=[i+1 for i,x in enumerate(metric['per_class_percent']) if x==0]
    return {'single_class_collapse':max(metric['prediction_shares'])>=.95,
            'zero_recall_class_count':len(zeros),'zero_recall_classes_one_based':zeros}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+',choices=('pavia','shanghai'),default=['pavia','shanghai'])
    args=parser.parse_args()
    summary={};lines=['# MLUDA datasets: frozen BiDA-style extension','',
        'Fixed epoch 200 only. Not MLUDA-protocol reproduction. No target-best selection.',
        'Historical MLUDA scores are not merged or used for cross-protocol gains.','']
    for task in args.tasks:
        rows={'B':[],'C':[]};paired=[];details=[]
        for seed in (2100,2101,2102):
            paths={arm:(c.ROOT/'results/pavia_joint_v1'/('joint_%d'%seed)
                        if task=='pavia' and arm=='B' else c.OUT/task/('%s_%d'%(arm,seed)))
                   for arm in ('B','C')}
            histories={arm:[json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()]
                       for arm,path in paths.items()}
            assert all(len(h)==200 and h[-1]['epoch']==200 for h in histories.values())
            for b,d in zip(histories['B'],histories['C']):
                for key in ('epoch','steps','batch_sha256','rng_sha256'): assert b[key]==d[key],(task,seed,key)
            results={arm:json.loads((path/'results.json').read_text()) for arm,path in paths.items()}
            for arm,result in results.items():
                assert c.helper.file_hash(paths[arm]/'frozen_manifest.json')==result['frozen_manifest_sha256']
                assert c.helper.file_hash(paths[arm]/'predictions_epoch_200.npz')==result['prediction']['sha256']
                metric=result['target_at_final'];rows[arm].append(metric)
                details.append({'seed':seed,'arm':arm,**metric,**profile(metric),
                    'source_val':result['source_val_at_final'],
                    'parameters':result['config']['trainable_parameters'],
                    'train_seconds':result['train_seconds'],'result_path':str(paths[arm])})
            paired.append({'seed':seed,**{k:results['C']['target_at_final'][k]-results['B']['target_at_final'][k]
                                         for k in ('oa_percent','aa_percent','kappa')}})
        aggregates={arm:stats(values) for arm,values in rows.items()}
        summary[task]={'details':details,'aggregate':aggregates,'paired_C_minus_B':paired,
                       'paired_aggregate':stats(paired),'all_200_epochs_data_and_rng_streams_equal':True,
                       'single_class_collapse_counts':{arm:sum(d['single_class_collapse'] for d in details if d['arm']==arm)
                                                       for arm in ('B','C')}}
        lines += ['## '+task,'','| Arm | OA mean±SD | AA mean±SD | Kappa mean±SD | Single-class collapse |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for arm in ('B','C'):
            a=aggregates[arm]
            values=['%.2f ± %.2f'%(a[k]['mean'],a[k]['sample_std']) for k in ('oa_percent','aa_percent')]
            values+=['%.4f ± %.4f'%(a['kappa']['mean'],a['kappa']['sample_std'])]
            lines.append('| %s | %s | %s | %s | %d/3 |'%(arm,*values,summary[task]['single_class_collapse_counts'][arm]))
        lines += ['','| Seed | C−B OA (pp) | C−B AA (pp) | C−B Kappa |','| --- | ---: | ---: | ---: |']
        for d in paired: lines.append('| %d | %+.3f | %+.3f | %+.5f |'%(d['seed'],d['oa_percent'],d['aa_percent'],d['kappa']))
        lines += ['','| Seed / Arm | Per-class recall (%) | Zero-recall classes | Parameters | Train seconds | Source-val OA |',
                  '| --- | --- | --- | ---: | ---: | ---: |']
        for d in details:
            lines.append('| %d / %s | %s | %s | %d | %.1f | %.2f |'%(d['seed'],d['arm'],
                ', '.join('%.2f'%x for x in d['per_class_percent']),str(d['zero_recall_classes_one_based']),
                d['parameters'],d['train_seconds'],d['source_val']['oa_percent']))
        lines += ['','Prediction distributions, paired aggregates and full metrics are in summary.json.',
                  'Dataset indices and actual batch/RNG streams match across arms for all 200 epochs.','']
    pavia_only=args.tasks==['pavia']
    c.dump(c.OUT/('pavia_summary.json' if pavia_only else 'summary.json'),summary)
    (c.OUT/('PAVIA_RESULTS.md' if pavia_only else 'RESULTS.md')).write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
