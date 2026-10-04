"""Score frozen seed2100 variants once and report, never select a version."""
import json
import numpy as np
from scipy.io import loadmat
import common as c


def main():
    manifests={}
    for arm in ('B','C'):
        folder=c.OUT/(arm+'_2100');path=folder/'frozen_manifest.json'
        manifest=json.loads(path.read_text())
        assert c.helper.file_hash(folder/'predictions_epoch_200.npz')==manifest['prediction']['sha256']
        manifests[arm]=manifest
        # Both prediction artifacts must already be frozen before reading GT values.
    raw=loadmat(c.DATA/'pavia_gt_7.mat')['pavia_gt_7'].astype(int)-1
    indices=np.argwhere(raw>=0);indices=indices[(indices[:,0]>0)&(indices[:,1]>0)]
    labels=raw[indices[:,0],indices[:,1]]
    rows={};paired={};histories={}
    for arm in ('B','C'):
        folder=c.OUT/(arm+'_2100');manifest=manifests[arm]
        history=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
        assert len(history)==200 and all(r['historical_normband_order_and_RNG_equal'] for r in history)
        histories[arm]=history
        assert len(labels)==manifest['config']['sample_counts']['target_eval']==39348
        with np.load(folder/'predictions_epoch_200.npz') as prediction:
            metrics=c.helper.metrics(labels,prediction['predictions'],prediction['logits'],7)
        zeros=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0]
        metrics.update(single_class_collapse=max(metrics['prediction_shares'])>=.95,
                       zero_recall_class_count=len(zeros),zero_recall_classes_one_based=zeros)
        result={**manifest,'target_at_final':metrics,'source_val_at_final':history[-1]['source_val'],
                'frozen_manifest_sha256':c.helper.file_hash(folder/'frozen_manifest.json')}
        c.dump(folder/'results.json',result)
        old=json.loads((c.reference(arm)/'results.json').read_text())
        rows[arm]={'zscore':metrics,'normband':old['target_at_final'],
                   'parameters':manifest['config']['trainable_parameters'],
                   'train_seconds':manifest['train_seconds'],'source_val':result['source_val_at_final'],
                   'reference_path':str(c.reference(arm))}
        paired[arm]={k:metrics[k]-old['target_at_final'][k] for k in ('oa_percent','aa_percent','kappa')}
    for b,d in zip(histories['B'],histories['C']):
        for key in ('epoch','steps','rng_sha256','batch_sha256','normband_shadow_batch_sha256','actual_dataset_indices_sha256'):
            assert b[key]==d[key],key
    cross={k:rows['C']['zscore'][k]-rows['B']['zscore'][k] for k in ('oa_percent','aa_percent','kappa')}
    summary={'seed':2100,'epoch':200,'interpretation':'single-seed normalization mechanism diagnostic only',
             'runs':rows,'zscore_minus_normband':paired,'zscore_C_minus_B':cross,
             'all_200_epochs_B_C_and_historical_sample_augmentation_RNG_match':True}
    c.dump(c.OUT/'summary.json',summary)
    lines=['# Pavia normalization diagnostic: seed2100, fixed200','',
           'Single-seed mechanism diagnostic; not MLUDA reproduction, no target-based version selection.',
           'Only input normalization changed. Source split/augmentation/initialization/updates unchanged.',
           '', '| Tokenizer | Normband OA | Z-score OA | ΔOA (pp) | Normband AA | Z-score AA | ΔAA (pp) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('B','C'):
        old,new=rows[arm]['normband'],rows[arm]['zscore'];delta=paired[arm]
        lines.append('| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f |'%(arm,old['oa_percent'],new['oa_percent'],
            delta['oa_percent'],old['aa_percent'],new['aa_percent'],delta['aa_percent']))
    lines += ['','| Tokenizer | Z-score Kappa | ΔKappa | Single-class collapse | Zero-recall classes | Parameters | Train seconds | Source-val OA |',
              '| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for arm in ('B','C'):
        row=rows[arm];m=row['zscore']
        lines.append('| %s | %.5f | %+.5f | %s | %s | %d | %.1f | %.3f |'%(arm,m['kappa'],paired[arm]['kappa'],
            m['single_class_collapse'],m['zero_recall_classes_one_based'],row['parameters'],row['train_seconds'],row['source_val']['oa_percent']))
    for arm in ('B','C'):
        m=rows[arm]['zscore']
        lines += ['','%s recall: '%arm+', '.join('%.2f'%v for v in m['per_class_percent']),
                  '%s prediction shares (%%): '%arm+', '.join('%.2f'%(v*100) for v in m['prediction_shares'])]
    lines += ['','Z-score C−B: '+json.dumps(cross),
              'All epoch sample/augmentation shadow hashes and dropout RNG hashes match historical normband.',
              'Concurrent timings include audit overhead; not an isolated preprocessing speed benchmark.',
              'No seed standard deviation/stability claim and no automatic additional experiments.']
    (c.OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
