"""Post-freeze scoring and assignment audits. GT never feeds back into F0-F3."""
import csv
import json
import shutil
import numpy as np
from common import HERE,OUT,CACHE,DATA,SEEDS,KS,GROUPS,sha,ahash,dump

def writecsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def metrics(y,p):
    cm=np.bincount(7*y+p,minlength=49).reshape(7,7);n=len(y);count=cm.sum(1)
    recall=np.divide(np.diag(cm),count,out=np.zeros(7,float),where=count>0)
    po=np.trace(cm)/n;pe=float(cm.sum(0)@cm.sum(1))/n**2
    return {'OA':float(100*po),'AA':float(100*recall[count>0].mean()),
            'Kappa':float((po-pe)/(1-pe)) if pe!=1 else None,
            'recall_percent':(100*recall).tolist(),'confusion':cm.tolist()}

def support(counts,y,k):
    other=counts.copy();other[np.arange(len(y)),y]=-1
    own=counts[np.arange(len(y)),y]
    return own/k,(own-other.max(1))/k

def masks(y,pred):
    right=y==pred
    result={'all':np.ones(len(y),bool),'original_correct':right,'original_error':~right}
    for c in range(7):
        for name,m in [('all',np.ones(len(y),bool)),('original_correct',right),('original_error',~right)]:
            result[f'C{c+1}_{name}']=(y==c)&m
        for j in range(7):
            m=(y==c)&(pred==j)
            if c!=j and m.any():result[f'C{c+1}_to_C{j+1}']=m
    return result

def change(before,after,y,mask,prefix):
    b=before[mask]==y[mask];a=after[mask]==y[mask]
    gain=int((~b&a).sum());loss=int((b&~a).sum())
    return {prefix+'_before_accuracy':float(b.mean()),prefix+'_after_accuracy':float(a.mean()),
        prefix+'_accuracy_delta':float(a.mean()-b.mean()),prefix+'_wrong_to_correct_n':gain,prefix+'_correct_to_wrong_n':loss,
        prefix+'_recovery_fraction_of_subset':float(gain/len(b)),prefix+'_damage_fraction_of_subset':float(loss/len(b)),
        prefix+'_recovery_fraction_of_before_wrong':float(gain/(~b).sum()) if (~b).any() else None,
        prefix+'_damage_fraction_of_before_correct':float(loss/b.sum()) if b.any() else None}

def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else None

def fixtures():
    y=np.array([0,1,2]);p=np.array([0,2,2]);m=metrics(y,p)
    assert np.isclose(m['OA'],200/3) and np.isclose(m['AA'],200/3)
    assert np.isclose(m['Kappa'],.5)
    b=np.array([1,1,0]);r=change(b,p,y,np.ones(3,bool),'clf')
    assert r['clf_wrong_to_correct_n']==2 and r['clf_correct_to_wrong_n']==1

def main():
    fixtures()
    gate=json.loads((OUT/'ALL_UNSUPERVISED_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_constructor']
    assert not gate['oracle_G2_artifacts_opened_by_constructor']
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    # Only now import old oracle-derived reference; it cannot influence F construction.
    oracle_gate=json.loads((CACHE/'ORACLE_RESULTS_FROZEN.json').read_text());imported={}
    for seed in SEEDS:
        for name in ['G2_rawZ.npy','G2_oracle_readouts.npz']:
            rel=f'seed_{seed}/{name}';assert sha(CACHE/rel)==oracle_gate['files_sha256'][rel]
            dst=OUT/f'seed_{seed}'/('Oracle_G2_rawZ.npy' if name.endswith('.npy') else 'Oracle_G2_readouts.npz')
            shutil.copyfile(CACHE/rel,dst);assert sha(dst)==oracle_gate['files_sha256'][rel]
            imported[str(dst.relative_to(OUT))]=sha(dst)
    dump(OUT/'ORACLE_REFERENCE_IMPORTED.json',{'files_sha256':imported,
         'imported_only_after_all_unsupervised_stats_routing_features_predictions_frozen':True,
         'used_only_for_scoring_reference':True,'old_oracle_manifest_sha256':sha(CACHE/'ORACLE_RESULTS_FROZEN.json')})
    print('All F0-F3 hashes verified, Oracle G2 reference authenticated afterwards. Opening target GT for scoring.',flush=True)
    y=np.load(DATA/'target_labels.npy')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    rows,metricrows,routingrows,centroidrows,details=[],[],[],[],{}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';native=np.load(folder/'target_predictions.npy');subset=masks(y,native)
        outputs={g:np.load(folder/f'{g}_readouts.npz') for g in ['F0','F1','F2','F3']}
        old=np.load(folder/'Oracle_G2_readouts.npz')
        outputs['Oracle_G2']={'classifier_predictions':old['frozen_classifier_predictions'],
            'classifier_logits':old['frozen_classifier_logits'],**{f'source_counts_k{k}':old[f'source_counts_k{k}'] for k in KS},
            **{f'source_knn_predictions_k{k}':old[f'source_knn_predictions_k{k}'] for k in KS}}
        assert np.array_equal(outputs['F0']['classifier_predictions'],native)
        head=np.load(folder/'frozen_head.npz');z=np.load(folder/'target_rawZ.npy').astype(float)
        oracle_z=np.load(folder/'Oracle_G2_rawZ.npy')
        replay=oracle_z@head['weight'].astype(float).T+head['bias'].astype(float)
        assert abs(replay-outputs['Oracle_G2']['classifier_logits']).max()<1e-10
        stats=np.load(folder/'fixed_unlabeled_statistics.npz');p=np.load(folder/'original_posterior.npy')
        r3=np.load(folder/'F3_soft_routing.npy')
        true_target=np.stack([z[y==c].mean(0) for c in range(7)])
        assert np.allclose(oracle_z,z+stats['source_means'][y]-true_target[y],atol=1e-12)
        for c in range(7):
            hp=native==c;true=y==c;mass=float(p[:,c].sum())
            true_delta=stats['source_means'][c]-true_target[c]
            centroidrows.append({'seed':seed,'class':c+1,'true_target_count':int(true.sum()),
                'hard_assignment_count':int(hp.sum()),'soft_assignment_mass':mass,
                'hard_assignment_true_class_purity':float((y[hp]==c).mean()) if hp.any() else None,
                'soft_assignment_true_class_purity':float(p[true,c].sum()/mass) if mass else None,
                'hard_target_centroid_L2_error':float(np.linalg.norm(stats['hard_target_means'][c]-true_target[c])),
                'soft_target_centroid_L2_error':float(np.linalg.norm(stats['soft_target_means'][c]-true_target[c])),
                'oracle_translation_norm':float(np.linalg.norm(true_delta)),
                'hard_translation_norm':float(np.linalg.norm(stats['hard_translation'][c])),
                'soft_translation_norm':float(np.linalg.norm(stats['soft_translation'][c])),
                'hard_vs_oracle_translation_cosine':cosine(stats['hard_translation'][c],true_delta),
                'soft_vs_oracle_translation_cosine':cosine(stats['soft_translation'][c],true_delta)})
        routes={'F1':np.eye(7)[native],'F2':p,'F3':r3}
        for group,routing in routes.items():
            np.savez_compressed(folder/f'{group}_routing_GT_audit.npz',true_class_routing_weight=routing[np.arange(len(y)),y],
                                routing_predictions=routing.argmax(1),target_labels=y)
            for name,mask in subset.items():
                if not mask.any():continue
                routingrows.append({'seed':seed,'group':group,'subset':name,'n':int(mask.sum()),
                    'true_class_routing_weight':float(routing[np.flatnonzero(mask),y[mask]].mean()),
                    'routing_argmax_correct_fraction':float((routing[mask].argmax(1)==y[mask]).mean()),
                    'routing_entropy_mean':float((-routing[mask]*np.log(np.maximum(routing[mask],np.finfo(float).tiny))).sum(1).mean())})
        details[str(seed)]={}
        for group in GROUPS:
            out=outputs[group];clf=out['classifier_predictions'];clfm=metrics(y,clf)
            details[str(seed)][group]={'classifier':clfm}
            for k in KS:
                knn=out[f'source_knn_predictions_k{k}'];knnm=metrics(y,knn)
                s,m= support(out[f'source_counts_k{k}'],y,k)
                bs,bm=support(outputs['F0'][f'source_counts_k{k}'],y,k)
                metricrows.append({'seed':seed,'group':group,'k_source':k,
                    'classifier_OA':clfm['OA'],'classifier_AA':clfm['AA'],'classifier_Kappa':clfm['Kappa'],
                    'source_knn_OA':knnm['OA'],'source_knn_AA':knnm['AA'],'source_knn_Kappa':knnm['Kappa'],
                    'source_correct_support_micro':float(s.mean()),'source_correct_support_macro':float(np.mean([s[y==c].mean() for c in range(7)])),
                    'source_support_margin_micro':float(m.mean()),'source_support_margin_macro':float(np.mean([m[y==c].mean() for c in range(7)]))})
                details[str(seed)][group][f'knn{k}']=knnm
                for name,mask in subset.items():
                    n=int(mask.sum())
                    if not n:continue
                    rows.append({'seed':seed,'group':group,'k_source':k,'subset':name,'n':n,
                        'source_support_before':float(bs[mask].mean()),'source_support_after':float(s[mask].mean()),
                        'source_support_delta':float((s[mask]-bs[mask]).mean()),
                        'source_margin_before':float(bm[mask].mean()),'source_margin_after':float(m[mask].mean()),
                        'source_margin_delta':float((m[mask]-bm[mask]).mean()),
                        'source_strict_support_restored_n':int((mask&(bm<=0)&(m>0)).sum()),
                        'source_strict_support_lost_n':int((mask&(bm>0)&(m<=0)).sum()),
                        **change(native,clf,y,mask,'classifier'),
                        **change(outputs['F0'][f'source_knn_predictions_k{k}'],knn,y,mask,'source_knn')})
            print(seed,group,'classifier OA/AA/Kappa',clfm['OA'],clfm['AA'],clfm['Kappa'],flush=True)
    paired,gaps=[],[]
    for row in metricrows:
        base=next(r for r in metricrows if r['seed']==row['seed'] and r['group']=='F0' and r['k_source']==row['k_source'])
        oracle=next(r for r in metricrows if r['seed']==row['seed'] and r['group']=='Oracle_G2' and r['k_source']==row['k_source'])
        delta={'seed':row['seed'],'group':row['group'],'k_source':row['k_source']}
        for metric in row:
            if metric in ['seed','group','k_source']:continue
            delta['delta_'+metric]=row[metric]-base[metric]
            den=oracle[metric]-base[metric]
            gaps.append({'seed':row['seed'],'group':row['group'],'k_source':row['k_source'],'metric':metric,
                'F_minus_F0':row[metric]-base[metric],'Oracle_G2_minus_F0':den,
                'oracle_gap_fraction':float((row[metric]-base[metric])/den) if den!=0 else None})
        paired.append(delta)
    writecsv('per_seed_metrics.csv',metricrows);writecsv('paired_deltas_vs_F0.csv',paired);writecsv('oracle_gap_capture.csv',gaps)
    writecsv('subset_statistics.csv',rows)
    writecsv('per_class.csv',[r for r in rows if r['subset'].startswith('C') and '_to_' not in r['subset']])
    writecsv('confusion_pairs.csv',[r for r in rows if '_to_' in r['subset']])
    writecsv('routing_audit.csv',routingrows);writecsv('centroid_estimation_audit.csv',centroidrows)
    aggregates=[]
    for group in GROUPS:
        for k in KS:
            rr=[r for r in metricrows if r['group']==group and r['k_source']==k]
            for key in rr[0]:
                if key in ['seed','group','k_source']:continue
                values=[r[key] for r in rr]
                aggregates.append({'group':group,'k_source':k,'metric':key,'mean':float(np.mean(values)),
                                   'sample_std':float(np.std(values,ddof=1)),'values':values})
    writecsv('seed_aggregate.csv',aggregates)
    limits=['All unsupervised assignments/statistics/hypotheses/routing/readouts frozen before GT/oracle reference import.',
        'One-step F1hard/F2soft/F3soft-statistics hypothesis routing; no re-estimation or temperature/threshold/fusion selection.',
        'F3 scores hypothesis own-class raw logit, T1; linear head makes this a fixed class-bias reweighting.',
        'G2 reference imported afterwards; true-class routing/statistics are oracle and not a strict upper bound.',
        'Gap fractions are un-clipped ratios, negative/greater-than-one possible; not significance or causal attribution.',
        'Wrong assignments contaminate class moment estimates and routing; their effects are intertwined in this fixed test.',
        'Failure of these variants does not prove impossibility of unsupervised transport or class routing generally.',
        'Raw CLS preserves frozen classifier convention; cosine kNN normalizes separately, unchanged source2393 train references.',
        'One scene pair,3 optimization seeds; C4 only22 pixels. No training/backward/tuning/follow-up.']
    dump(OUT/'summary.json',{'metrics':metricrows,'subsets':rows,'paired_deltas':paired,'gap_capture':gaps,
        'routing_audit':routingrows,'centroid_estimation_audit':centroidrows,'aggregates':aggregates,
        'details':details,'limitations':limits})
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    for path,h in imported.items():assert sha(OUT/path)==h
    dump(OUT/'SCORING_COMPLETE.json',{'all_unsupervised_artifacts_checkpoints_unchanged':True,
        'GT_only_scoring_audits_after_full_unsupervised_freeze':True,'oracle_reference_not_used_for_construction':True,
        'no_training_backward_tuning_followup':True,'summary_sha256':sha(OUT/'summary.json'),
        'target_labels_sha256':sha(DATA/'target_labels.npy'),'fixtures_passed':True})

if __name__=='__main__':main()
