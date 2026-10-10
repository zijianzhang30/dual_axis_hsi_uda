"""Offline interpretation of previously frozen consensus predictions; no tuning."""
import csv
import json
import time
import numpy as np
from common import HERE,OUT,DATA,SEEDS,STAGES,MODES,KS,sha,ahash,dump

def counts_support(counts,y,k):
    wrong=counts.copy();wrong[np.arange(len(y)),y]=-1
    own=counts[np.arange(len(y)),y]
    return own/k,(own-wrong.max(1))/k

def subsets(y,pred):
    correct=y==pred
    masks={'all':np.ones(len(y),bool),'original_correct':correct,'original_error':~correct}
    for c in range(7):
        for name,m in [('all',np.ones(len(y),bool)),('original_correct',correct),('original_error',~correct)]:
            masks[f'C{c+1}_{name}']=(y==c)&m
        for j in range(7):
            if j!=c and ((y==c)&(pred==j)).any():masks[f'C{c+1}_to_C{j+1}']=(y==c)&(pred==j)
    return masks

def ratio(n,d):return float(n/d) if d else None

def classification_changes(before,after,y,mask,prefix):
    b=before[mask]==y[mask];a=after[mask]==y[mask]
    recover=int((~b&a).sum());damage=int((b&~a).sum())
    return {prefix+'_accuracy_before':float(b.mean()),prefix+'_accuracy_after':float(a.mean()),
            prefix+'_accuracy_delta':float(a.mean()-b.mean()),prefix+'_wrong_to_correct_n':recover,
            prefix+'_wrong_to_correct_fraction':ratio(recover,len(b)),
            prefix+'_wrong_to_correct_fraction_of_before_wrong':ratio(recover,int((~b).sum())),
            prefix+'_correct_to_wrong_n':damage,prefix+'_correct_to_wrong_fraction':ratio(damage,len(b)),
            prefix+'_correct_to_wrong_fraction_of_before_correct':ratio(damage,int(b.sum()))}

def describe_change(before,after,mask,prefix):
    b=before[mask];a=after[mask]
    return {prefix+'_before':float(b.mean()),prefix+'_after':float(a.mean()),prefix+'_delta':float((a-b).mean()),
            prefix+'_before_q05':float(np.quantile(b,.05)),prefix+'_after_q05':float(np.quantile(a,.05))}

def subset_metrics(y,mask,b,a,bp,ap,bc,ac,purity,overlap):
    n=int(mask.sum())
    if not n:return {'n':0}
    before_bad=b[1][mask]<=0;after_good=a[1][mask]>0
    restored=int((before_bad&after_good).sum())
    lost=int((~before_bad&~after_good).sum())
    row={'n':n,**describe_change(b[0],a[0],mask,'source_correct_support'),
         **describe_change(b[1],a[1],mask,'source_support_margin'),
         'source_insufficient_before_fraction':float(before_bad.mean()),
         'source_insufficient_after_fraction':float((~after_good).mean()),
         'source_support_restored_n':restored,'source_support_restored_fraction':ratio(restored,n),
         'source_support_restored_fraction_of_before_insufficient':ratio(restored,int(before_bad.sum())),
         'source_support_lost_n':lost,'source_support_lost_fraction':ratio(lost,n),
         'source_support_lost_fraction_of_before_strict_support':ratio(lost,int((~before_bad).sum())),
         'source_margin_tie_before_fraction':float((b[1][mask]==0).mean()),
         'source_margin_tie_after_fraction':float((a[1][mask]==0).mean()),
         'target_neighbor_purity':float(purity[mask].mean()),
         'target_neighbor_overlap_fraction':float(overlap[mask].mean())}
    row.update(classification_changes(bp,ap,y,mask,'source_knn'))
    row.update(classification_changes(bc,ac,y,mask,'source_centroid'))
    return row

def writecsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def fixtures():
    y=np.array([0,1,2,3]);bp=np.array([1,1,3,3]);ap=np.array([0,2,2,3])
    b=(np.array([0,.9,0,.9]),np.array([-1,.8,-1,.8]))
    a=(np.array([.9,0,.9,.9]),np.array([.8,-1,.8,.8]))
    r=subset_metrics(y,np.ones(4,bool),b,a,bp,ap,bp,ap,np.ones(4),np.zeros(4))
    assert r['source_support_restored_n']==2 and r['source_support_lost_n']==1
    assert r['source_knn_wrong_to_correct_n']==2 and r['source_knn_correct_to_wrong_n']==1
    assert r['source_support_restored_fraction_of_before_insufficient']==1
    assert r['source_knn_wrong_to_correct_fraction_of_before_wrong']==1
    assert r['source_knn_accuracy_delta']==.25

def main():
    start=time.monotonic();fixtures()
    gate=json.loads((OUT/'ALL_COUNTERFACTUALS_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and gate['features_neighbors_predictions_all_frozen']
    assert not gate['target_GT_opened_by_constructor']
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    provenance=json.loads((OUT/'provenance.json').read_text())
    for path,h in provenance['code_sha256'].items():assert sha(path)==h
    assert provenance['protocol_sha256']==sha(HERE/'PROTOCOL.txt')
    print('All seed features/neighbors/counterfactual predictions verified; opening target GT offline.',flush=True)
    y=np.load(DATA/'target_labels.npy');sy=np.load(OUT/'source_labels.npy')
    assert ahash(y)==provenance['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    rows,macro,seedmeta=[],[],{}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';native=np.load(folder/'target_predictions.npy')
        logits=np.load(folder/'target_logits.npy');assert np.array_equal(native,logits.argmax(1))
        assert np.array_equal(coords,np.load(folder/'target_coords.npy'))
        maskset=subsets(y,native);cm=np.bincount(7*y+native,minlength=49).reshape(7,7)
        seedmeta[str(seed)]={'native_confusion':cm.tolist(),'original_error_n':int((native!=y).sum()),
                             'native_classifier_used_only_for_fixed_subsets':True}
        for stage in STAGES:
            before=np.load(folder/f'{stage}_before_readouts.npz')
            for mode in MODES:
                neighbors=np.load(folder/f'{stage}_{mode}_target_neighbors.npz')
                ni=neighbors['indices'];dist=neighbors['center_Chebyshev_distance']
                assert not (ni==np.arange(len(y))[:,None]).any()
                if mode=='nonoverlap_knn':assert (dist>12).all()
                for kt in KS:
                    purity=(y[ni[:,:kt]]==y[:,None]).mean(1)
                    overlap=(dist[:,:kt]<=12).mean(1)
                    after=np.load(folder/f'{stage}_{mode}_kt{kt}_readouts.npz')
                    for ks in KS:
                        bcounts=before[f'source_counts_k{ks}'];acounts=after[f'source_counts_k{ks}']
                        assert np.array_equal(bcounts.argmax(1),before[f'source_knn_predictions_k{ks}'])
                        assert np.array_equal(acounts.argmax(1),after[f'source_knn_predictions_k{ks}'])
                        b=counts_support(bcounts,y,ks);a=counts_support(acounts,y,ks)
                        bp=before[f'source_knn_predictions_k{ks}'];ap=after[f'source_knn_predictions_k{ks}']
                        bc=before['centroid_predictions'];ac=after['centroid_predictions']
                        np.savez_compressed(folder/f'{stage}_{mode}_kt{kt}_ks{ks}_offline_signals.npz',
                            labels=y,native_predictions=native,source_support_before=b[0],source_support_after=a[0],
                            source_margin_before=b[1],source_margin_after=a[1],target_neighbor_purity=purity,
                            target_neighbor_overlap_fraction=overlap,source_support_restored=(b[1]<=0)&(a[1]>0),
                            source_support_lost=(b[1]>0)&(a[1]<=0),source_knn_wrong_to_correct=(bp!=y)&(ap==y),
                            source_knn_correct_to_wrong=(bp==y)&(ap!=y),source_centroid_wrong_to_correct=(bc!=y)&(ac==y),
                            source_centroid_correct_to_wrong=(bc==y)&(ac!=y))
                        strata={}
                        for name,mask in maskset.items():
                            r=subset_metrics(y,mask,b,a,bp,ap,bc,ac,purity,overlap)
                            if not r['n']:continue
                            strata[name]=r
                            rows.append({'seed':seed,'stage':stage,'mode':mode,'k_target':kt,'k_source':ks,'subset':name,**r})
                        for subset in ['all','original_correct','original_error']:
                            rr=[strata[f'C{c+1}_{subset}'] for c in range(7) if f'C{c+1}_{subset}' in strata]
                            m={'seed':seed,'stage':stage,'mode':mode,'k_target':kt,'k_source':ks,'subset':subset,'represented_classes':len(rr)}
                            for key in rr[0]:
                                if key=='n' or key.endswith('_n'):continue
                                vals=[r[key] for r in rr if r[key] is not None]
                                m[key]=float(np.mean(vals)) if vals else None
                            macro.append(m)
                    print(seed,stage,mode,'kt',kt,'offline scored',flush=True)
    writecsv('subset_statistics.csv',rows)
    writecsv('per_seed.csv',[r for r in rows if r['subset'] in ['all','original_correct','original_error']])
    writecsv('per_class.csv',[r for r in rows if r['subset'].startswith('C') and '_to_' not in r['subset']])
    writecsv('confusion_pairs.csv',[r for r in rows if '_to_' in r['subset']])
    writecsv('class_macro_per_seed.csv',macro)
    lookup={(r['seed'],r['stage'],r['mode'],r['k_target'],r['k_source'],r['subset']):r for r in rows}
    audit=[]
    for key,s in lookup.items():
        seed,stage,mode,kt,ks,subset=key
        if mode!='standard_knn':continue
        n=lookup[(seed,stage,'nonoverlap_knn',kt,ks,subset)]
        a={'seed':seed,'stage':stage,'k_target':kt,'k_source':ks,'subset':subset,'n':s['n']}
        for metric in ['target_neighbor_purity','target_neighbor_overlap_fraction','source_correct_support_delta',
                       'source_support_margin_delta','source_support_restored_fraction',
                       'source_support_restored_fraction_of_before_insufficient','source_support_lost_fraction',
                       'source_knn_accuracy_delta','source_knn_wrong_to_correct_fraction','source_knn_correct_to_wrong_fraction']:
            a[metric+'_standard']=s[metric];a[metric+'_nonoverlap']=n[metric]
            a[metric+'_nonoverlap_minus_standard']=n[metric]-s[metric] if n[metric] is not None and s[metric] is not None else None
        audit.append(a)
    writecsv('neighbor_overlap_audit.csv',audit)
    aggregates=[]
    for stage in STAGES:
        for mode in MODES:
            for kt in KS:
                for ks in KS:
                    names=sorted({r['subset'] for r in rows if r['stage']==stage and r['mode']==mode and r['k_target']==kt and r['k_source']==ks})
                    for subset in names:
                        rr=[r for r in rows if r['stage']==stage and r['mode']==mode and r['k_target']==kt and r['k_source']==ks and r['subset']==subset]
                        for metric in rr[0]:
                            if metric in ['seed','stage','mode','k_target','k_source','subset']:continue
                            values=[r[metric] for r in rr if r[metric] is not None]
                            if values:aggregates.append({'stage':stage,'mode':mode,'k_target':kt,'k_source':ks,'subset':subset,
                                'metric':metric,'available_seeds':len(values),'mean':float(np.mean(values)),
                                'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None,
                                'seeds':[r['seed'] for r in rr if r[metric] is not None],'values':values})
    writecsv('seed_aggregate.csv',aggregates)
    limits=['Fixed full grid; no selecting best k/stage/mode. Source references unchanged.',
            'Consensus uses feature-only target neighbors; target GT is offline scoring only.',
            'nonoverlap excludes Chebyshev<=12, including self; patch13 pixel footprints do not intersect.',
            'Nonoverlap does not remove all scene autocorrelation, nearby materials, or correlated queries.',
            'Source votes reflect class composition; equal votes use lowest class index, strict support excludes ties.',
            'Native raw-CLS classifier is not applied to unit consensus Z; reported readouts are frozen source centroid/kNN.',
            'Original classifier correctness defines subsets, not baseline kNN correctness; all denominators are explicit.',
            'Counterfactual gains/damage are descriptive, not learned UDA method accuracy or causal attribution.',
            'One scene pair, three optimization seeds; C4 only22 pixels. Micro and class macro reported separately.']
    dump(OUT/'summary.json',{'per_seed_rows':rows,'class_macro_rows':macro,'seed_metadata':seedmeta,
         'seed_aggregates':aggregates,'neighbor_overlap_audit':audit,'limitations':limits,
         'fixture_checks_passed':True,'seconds':time.monotonic()-start})
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    for rec in provenance['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'counterfactual_artifacts_checkpoints_unchanged':True,
        'target_GT_first_opened_after_all_counterfactual_freeze':True,'no_training_backward_tuning_followup':True,
        'summary_sha256':sha(OUT/'summary.json'),'target_labels_sha256':sha(DATA/'target_labels.npy')})

if __name__=='__main__':main()
