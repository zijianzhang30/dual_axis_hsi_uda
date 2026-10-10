"""Fixed descriptive kNN interpretation; no fitted model or selected parameters."""
import csv
import json
import time
import numpy as np
from common import HERE, OUT, BASE, DATA, SEEDS, STAGES, KS, dump, sha, ahash

SIGNALS=('source_correct_support','source_support_margin','target_true_class_purity',
         'target_plurality_margin','native_logit_margin')
CATEGORIES=('source-local-support error','target-local-mixing','classifier-boundary disagreement')

def class_counts(labels, ids, k):
    picked=labels[ids[:,:k]]
    return np.stack([(picked==c).sum(1) for c in range(7)],axis=1)

def support(counts,y,k):
    wrong=counts.copy();wrong[np.arange(len(y)),y]=-1
    competitor=wrong.argmax(1)
    own=counts[np.arange(len(y)),y]
    return own/k,(own-wrong.max(1))/k,competitor

def sample_signals(sc,tc,y,logits,pred,k):
    sp,sm,scomp=support(sc,y,k)
    tp,tm,tcomp=support(tc,y,k)
    wrong=logits.copy();wrong[np.arange(len(y)),y]=-np.inf
    competitor=wrong.argmax(1)
    lm=logits[np.arange(len(y)),y]-wrong.max(1)
    error=pred!=y
    source_bad=sm<=0;target_bad=tm<=0
    boundary=error&~source_bad&~target_bad
    cat=np.full(len(y),-1,dtype=np.int8)
    cat[error&source_bad]=0
    cat[error&~source_bad&target_bad]=1
    cat[boundary]=2
    assert (cat[error]>=0).all() and (cat[~error]==-1).all()
    return {'source_correct_support':sp,'source_support_margin':sm,'target_true_class_purity':tp,
            'target_plurality_margin':tm,'native_logit_margin':lm,'source_competitor':scomp,
            'target_competitor':tcomp,'native_logit_competitor':competitor,
            'source_local_support_error':source_bad,'target_local_mixing':target_bad,
            'classifier_boundary_disagreement':boundary,'source_tie':sm==0,'target_tie':tm==0,
            'exclusive_error_category':cat,'native_error':error}

def describe(x):
    return {'mean':float(x.mean()),'median':float(np.median(x)),
            'q05':float(np.quantile(x,.05)),'q95':float(np.quantile(x,.95))}

def subset_masks(y,pred):
    correct=y==pred
    masks={'all':np.ones(len(y),bool),'correct':correct,'error':~correct}
    for c in range(7):
        for kind,mask in [('all',np.ones(len(y),bool)),('correct',correct),('error',~correct)]:
            masks[f'C{c+1}_{kind}']=(y==c)&mask
        for j in range(7):
            if j!=c:
                m=(y==c)&(pred==j)
                if m.any():masks[f'C{c+1}_to_C{j+1}']=m
    return masks

def subset_stats(signals,sc,tc,mask,k):
    n=int(mask.sum())
    if not n:return {'n':0}
    r={'n':n,'native_error_fraction':float(signals['native_error'][mask].mean())}
    for name in SIGNALS:
        r.update({name+'_'+key:v for key,v in describe(signals[name][mask]).items()})
    for name in ['source_local_support_error','target_local_mixing','classifier_boundary_disagreement','source_tie','target_tie']:
        r[name+'_fraction']=float(signals[name][mask].mean())
    for c in range(7):
        r[f'source_neighbor_C{c+1}_fraction']=float(sc[mask,c].mean()/k)
        r[f'target_neighbor_C{c+1}_fraction']=float(tc[mask,c].mean()/k)
    error=mask&signals['native_error'];ne=int(error.sum());r['native_error_n']=ne
    for cat in range(3):
        count=int((mask&(signals['exclusive_error_category']==cat)).sum())
        r[f'category_{cat}_n']=count
        r[f'category_{cat}_fraction_of_error']=count/ne if ne else None
    # Independent source/target flags, without prioritizing one as causal.
    for s in range(2):
        for t in range(2):
            count=int((error&(signals['source_local_support_error']==bool(s))&(signals['target_local_mixing']==bool(t))).sum())
            r[f'error_overlap_source{s}_target{t}_n']=count
    return r

def writecsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def checks():
    y=np.array([0,1,2,3]);pred=np.array([1,1,3,4])
    sc=np.array([[0,5,0,0,0,0,0],[0,4,1,0,0,0,0],[0,0,4,1,0,0,0],[0,0,0,4,1,0,0]])
    tc=np.array([[5,0,0,0,0,0,0],[0,4,1,0,0,0,0],[0,0,1,4,0,0,0],[0,0,0,4,1,0,0]])
    logits=np.eye(7)[pred]*3
    s=sample_signals(sc,tc,y,logits,pred,5)
    assert s['exclusive_error_category'].tolist()==[0,-1,1,2]
    labels=np.array([0,1,1,2,2,2]);ids=np.array([[0,1,2,3,4],[1,2,3,4,5]])
    c=class_counts(labels,ids,5)
    assert np.array_equal(c,np.array([[1,2,2,0,0,0,0],[0,2,3,0,0,0,0]]))
    sp,sm,_=support(c,np.array([1,2]),5)
    assert np.allclose(sp,[.4,.6]) and np.allclose(sm,[0,.2])

def main():
    start=time.monotonic();checks()
    gate=json.loads((OUT/'ALL_NEIGHBORS_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_this_diagnostic']
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    provenance=json.loads((OUT/'provenance.json').read_text())
    assert sha(HERE/'PROTOCOL.txt')==provenance['protocol_sha256']
    for path,h in provenance['code_sha256'].items():assert sha(path)==h
    print('All3 seed features, logits, predictions and exact neighbors verified. Opening labels offline.',flush=True)
    sy=np.load(DATA/'source_labels.npy');ty=np.load(DATA/'target_labels.npy')
    for domain,labels in [('source',sy),('target',ty)]:
        assert ahash(labels)==provenance['historical_metadata']['domains'][domain]['labels_sha256']
        coords=np.load(OUT/'seed_2100'/f'{domain}_coords.npy')
        gt=np.load(DATA/f'{domain}_gt.npy',mmap_mode='r')
        assert np.array_equal(labels,gt[coords[:,0],coords[:,1]])
    assert np.array_equal(ty,np.load(BASE/'houston/evaluation_labels.npy'))
    train=np.load(OUT/'source_train_mask.npy')
    runs,rows,comparison,transitionrows,macrorows={ },[],[],[],[]
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';logits=np.load(folder/'target_logits.npy').astype(np.float64)
        pred=np.load(folder/'target_predictions.npy');assert np.array_equal(pred,logits.argmax(1))
        masks=subset_masks(ty,pred);cm=np.bincount(7*ty+pred,minlength=49).reshape(7,7)
        pairs=sorted([(int(cm[c,j]),c+1,j+1) for c in range(7) for j in range(7) if c!=j and cm[c,j]],key=lambda v:(-v[0],v[1],v[2]))
        runs[str(seed)]={'native_confusion':cm.tolist(),'native_OA_percent':float(100*(pred==ty).mean()),
                         'native_AA_percent':float(np.mean([100*(pred[ty==c]==c).mean() for c in range(7)])),
                         'top5_native_confusion_pairs':[{'n':n,'true':c,'predicted':j} for n,c,j in pairs[:5]],'stages':{}}
        cached={}
        for stage in STAGES:
            nb=np.load(folder/f'{stage}_neighbors.npz')
            assert train[nb['source_indices']].all()
            assert not (nb['target_indices']==np.arange(len(ty))[:,None]).any()
            runs[str(seed)]['stages'][stage]={}
            for k in KS:
                sc=class_counts(sy,nb['source_indices'],k);tc=class_counts(ty,nb['target_indices'],k)
                sig=sample_signals(sc,tc,ty,logits,pred,k);cached[(stage,k)]=sig
                np.savez_compressed(folder/f'{stage}_k{k}_offline_signals.npz',source_counts=sc,target_counts=tc,
                                    target_labels=ty,native_predictions=pred,**sig)
                strata={name:subset_stats(sig,sc,tc,mask,k) for name,mask in masks.items()}
                runs[str(seed)]['stages'][stage][str(k)]={'subsets':strata}
                for name,r in strata.items():
                    if r['n']:rows.append({'seed':seed,'stage':stage,'k':k,'subset':name,**r})
                for kind in ['all','correct','error']:
                    cr=[strata[f'C{c+1}_{kind}'] for c in range(7) if strata[f'C{c+1}_{kind}']['n']]
                    macro={'seed':seed,'stage':stage,'k':k,'subset':kind,'represented_classes':len(cr)}
                    for key in cr[0]:
                        if key not in ['n','native_error_n'] and not key.endswith('_n'):
                            values=[r[key] for r in cr if r[key] is not None]
                            macro[key]=float(np.mean(values)) if values else None
                    macrorows.append(macro)
                print(seed,stage,'k',k,'error source-support',strata['error']['source_local_support_error_fraction'],
                      'target mixing',strata['error']['target_local_mixing_fraction'],
                      'boundary',strata['error']['classifier_boundary_disagreement_fraction'],flush=True)
        for k in KS:
            a,b=cached[('T',k)],cached[('Z',k)]
            assert np.array_equal(a['native_logit_margin'],b['native_logit_margin'])
            for name,mask in masks.items():
                if not mask.any():continue
                row={'seed':seed,'k':k,'subset':name,'n':int(mask.sum())}
                for signal in SIGNALS:
                    row[f'delta_{signal}_mean']=float((b[signal][mask]-a[signal][mask]).mean())
                for flag in ['source_local_support_error','target_local_mixing','classifier_boundary_disagreement']:
                    row[f'delta_{flag}_fraction']=float(b[flag][mask].mean()-a[flag][mask].mean())
                    row[f'{flag}_resolved_n']=int((mask&a[flag]&~b[flag]).sum())
                    row[f'{flag}_introduced_n']=int((mask&~a[flag]&b[flag]).sum())
                comparison.append(row)
                error=mask&a['native_error']
                if error.any():
                    mat=np.bincount(3*a['exclusive_error_category'][error]+b['exclusive_error_category'][error],minlength=9).reshape(3,3)
                    for i in range(3):
                        for j in range(3):transitionrows.append({'seed':seed,'k':k,'subset':name,'T_category':CATEGORIES[i],'Z_category':CATEGORIES[j],'n':int(mat[i,j]),'fraction_of_subset_error':float(mat[i,j]/error.sum())})
    writecsv('subset_statistics.csv',rows);writecsv('macro_statistics.csv',macrorows)
    writecsv('T_to_Z_signal_changes.csv',comparison);writecsv('T_to_Z_error_category_transitions.csv',transitionrows)
    # Aggregate same strata across optimization seeds; empty subsets remain absent, not zero-filled.
    aggrows=[]
    for stage in STAGES:
        for k in KS:
            names=sorted({r['subset'] for r in rows if r['stage']==stage and r['k']==k})
            for name in names:
                rr=[r for r in rows if r['stage']==stage and r['k']==k and r['subset']==name]
                for key in [k for k in rr[0] if k.endswith('_mean') or k.endswith('_fraction') or k.endswith('_fraction_of_error')]:
                    vals=[r[key] for r in rr if r[key] is not None]
                    if vals:aggrows.append({'stage':stage,'k':k,'subset':name,'metric':key,'available_seeds':len(vals),
                                           'mean':float(np.mean(vals)),'sample_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'values':vals})
    writecsv('subset_seed_aggregate.csv',aggrows)
    limits=['No fitting, tuning, new model or follow-up experiment.',
            'Source2393 train only; target52901; exact cosine kNN with k5/15 and target self-exclusion.',
            'Original Z-classifier outcome and logit margin are shared fixed references for T/Z.',
            'Flags and exclusive partition describe disagreement; no causal attribution.',
            'Ties are ambiguous support and included in <=0 flags; reported separately.',
            'Target LOO retains spatial/patch overlap dependence; high purity need not imply transferability.',
            'Source class imbalance affects neighbor composition; votes are not calibrated probabilities.',
            'Class macro differs from sample micro. C4 has22 target pixels; one transfer pair, three optimization seeds.']
    dump(OUT/'summary.json',{'runs':runs,'seed_aggregates':aggrows,'macro_rows':macrorows,'T_to_Z':comparison,
         'category_labels':CATEGORIES,'limitations':limits,'checks_passed':True,'seconds':time.monotonic()-start})
    report(runs,limits)
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    for rec in provenance['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_artifacts_and_checkpoints_unchanged':True,
         'target_GT_only_after_all_features_and_neighbors_frozen':True,'no_training_tuning_followup':True,
         'label_file_sha256':{str(DATA/f'{d}_labels.npy'):sha(DATA/f'{d}_labels.npy') for d in ['source','target']},
         'report_sha256':sha(OUT/'REPORT.txt'),'summary_sha256':sha(OUT/'summary.json')})

def report(runs,limits):
    lines=['Houston local support diagnostic V1; original Joint BN+C, seeds2100/2101/2102, epoch200.',
           'Independent authenticated feature/prediction copies and exact neighbors frozen before label scoring.',
           'All stages sample-L2 normalized. Source2393 train only; target52901, leave-one-out.',
           'k=5,15 unweighted cosine neighbors. Native classifier/logit margin comes from Z and is identical atT/Z.',
           'Source support error=correct votes <= best wrong votes. Target mixing defined the same way.',
           'Boundary disagreement=classifier wrong despite strict correct plurality in both neighborhoods.',
           'Exclusive error partition prioritizes source error, then target mixing, then boundary disagreement.',
           'Independent flags and overlap counts are also in CSV/NPZ. All findings are descriptive.', '']
    for subset in ['all','correct','error','C1_error','C7_error']:
        lines.append(subset+'; three-seed sample-micro means within each subset:')
        for k in KS:
            for stage in STAGES:
                rr=[runs[str(s)]['stages'][stage][str(k)]['subsets'][subset] for s in SEEDS]
                rr=[r for r in rr if r['n']]
                mean=lambda key:np.mean([r[key] for r in rr])
                lines.append(f'  {stage} k{k}: n={[r["n"] for r in rr]}; source support={mean("source_correct_support_mean"):.5f}, '
                             f'source margin={mean("source_support_margin_mean"):+.5f}, target purity={mean("target_true_class_purity_mean"):.5f}, '
                             f'target margin={mean("target_plurality_margin_mean"):+.5f}, native logit margin={mean("native_logit_margin_mean"):+.5f}')
                lines.append(f'    independent source-error={100*mean("source_local_support_error_fraction"):.2f}%, '
                             f'target-mixing={100*mean("target_local_mixing_fraction"):.2f}%, '
                             f'boundary={100*mean("classifier_boundary_disagreement_fraction"):.2f}%; '
                             f'source ties={100*mean("source_tie_fraction"):.2f}%, target ties={100*mean("target_tie_fraction"):.2f}%')
                if subset.endswith('error'):
                    lines.append('    exclusive fractions of errors: '+'; '.join(f'{CATEGORIES[i]}={100*mean(f"category_{i}_fraction_of_error"):.2f}%' for i in range(3)))
        lines.append('')
    lines.append('Native confusion pair counts (presentation only; every nonempty directed pair is in subset_statistics.csv):')
    for seed in SEEDS:
        lines.append(f'  seed{seed} top5: '+str(runs[str(seed)]['top5_native_confusion_pairs']))
    lines.extend(['','Limitations:',*limits])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
