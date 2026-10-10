"""GT-assisted oracle transforms after native input freeze; no model training."""
import csv
import importlib.util
import json
import time
import numpy as np
import torch
from common import HERE,OUT,DATA,SEEDS,GROUPS,KS,sha,ahash,unit,dump

helper_path=HERE.parent/'houston_local_support_diagnostic_v1/freeze_neighbors.py'
spec=importlib.util.spec_from_file_location('exact_neighbor_helper',helper_path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def moments(x):
    mu=x.mean(0);a=x-mu;cov=a.T@a/len(x)
    return mu,(cov+cov.T)/2

def matrix_power(x,p):
    e,v=np.linalg.eigh((x+x.T)/2)
    floor=max(np.finfo(np.float64).tiny,float(e.max())*np.finfo(np.float64).eps)
    return (v*np.maximum(e,floor)**p)@v.T

def affine(source,target):
    ms,cs=moments(source);mt,ct=moments(target);dim=source.shape[1]
    sr=1e-6*max(float(np.trace(cs)/dim),1e-12)
    tr=1e-6*max(float(np.trace(ct)/dim),1e-12)
    csr=cs+sr*np.eye(dim);ctr=ct+tr*np.eye(dim)
    root=matrix_power(ctr,.5);inverse=matrix_power(ctr,-.5)
    a=inverse@matrix_power(root@csr@root,.5)@inverse;a=(a+a.T)/2
    residual=float(np.linalg.norm(a@ctr@a-csr,'fro')/max(np.linalg.norm(csr,'fro'),1e-15))
    assert np.isfinite(a).all() and residual<1e-4
    ev=np.linalg.eigvalsh(a)
    audit={'source_n':len(source),'target_n':len(target),'source_cov_rank':int(np.linalg.matrix_rank(cs)),
           'target_cov_rank':int(np.linalg.matrix_rank(ct)),'source_cov_ridge':sr,'target_cov_ridge':tr,
           'regularized_covariance_relative_residual':residual,'affine_min_eigenvalue':float(ev.min()),
           'affine_max_eigenvalue':float(ev.max()),'affine_condition':float(ev.max()/ev.min()),
           'centroid_translation_norm':float(np.linalg.norm(ms-mt))}
    return a,ms,mt,audit

@torch.inference_mode()
def source_knn(q,source,source_ids):
    q=torch.tensor(unit(q),device='cuda',dtype=torch.float64)
    s=torch.tensor(unit(source),device='cuda',dtype=torch.float64)
    ids,values=[],[];ties=0
    for i in range(0,len(q),512):
        ix,vs,nt=helper.select_neighbors(q[i:i+512]@s.T)
        ids.append(source_ids[ix]);values.append(vs);ties+=nt
    return np.concatenate(ids),np.concatenate(values),ties

def metric(y,p):
    cm=np.bincount(7*y+p,minlength=49).reshape(7,7);counts=cm.sum(1)
    present=counts>0;recall=np.divide(np.diag(cm),counts,out=np.zeros(7,float),where=present)
    return {'OA_percent':float(100*np.trace(cm)/len(y)),'AA_percent':float(100*recall[present].mean()),
            'represented_classes':int(present.sum()),'recall_percent':(100*recall).tolist(),'confusion':cm.tolist()}

def margin(counts,y,k):
    other=counts.copy();other[np.arange(len(y)),y]=-1
    own=counts[np.arange(len(y)),y]
    return own/k,(own-other.max(1))/k

def masks(y,native):
    right=y==native
    result={'all':np.ones(len(y),bool),'original_correct':right,'original_error':~right}
    for c in range(7):
        for subset,m in [('all',np.ones(len(y),bool)),('original_correct',right),('original_error',~right)]:
            result[f'C{c+1}_{subset}']=(y==c)&m
        for j in range(7):
            if c!=j:
                m=(y==c)&(native==j)
                if m.any():result[f'C{c+1}_to_C{j+1}']=m
    return result

def classification_change(before,after,y,mask,prefix):
    b=before[mask]==y[mask];a=after[mask]==y[mask]
    gain=int((~b&a).sum());loss=int((b&~a).sum())
    return {prefix+'_accuracy_before':float(b.mean()),prefix+'_accuracy_after':float(a.mean()),
            prefix+'_accuracy_delta':float(a.mean()-b.mean()),prefix+'_recovered_n':gain,prefix+'_damaged_n':loss,
            prefix+'_recovered_fraction_of_subset':float(gain/len(b)),prefix+'_damaged_fraction_of_subset':float(loss/len(b)),
            prefix+'_recovered_fraction_of_before_wrong':float(gain/(~b).sum()) if (~b).any() else None,
            prefix+'_damaged_fraction_of_before_correct':float(loss/b.sum()) if b.any() else None}

def fixtures():
    rng=np.random.default_rng(2100);t=rng.normal(size=(200,9));s=2*t+3
    a,ms,mt,check=affine(s,t)
    assert np.allclose(a,2*np.eye(9),atol=1e-8)
    assert np.allclose((t-mt)@a+ms,s,atol=1e-8)
    translated=t+ms-mt
    assert np.allclose(translated.mean(0),s.mean(0))
    assert check['regularized_covariance_relative_residual']<1e-8
    y=np.array([0,1,2]);b=np.array([1,1,0]);p=np.array([0,2,2])
    r=classification_change(b,p,y,np.ones(3,bool),'classifier')
    assert r['classifier_recovered_n']==2 and r['classifier_damaged_n']==1

def writecsv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    start=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    fixtures();helper.check_knn()
    gate=json.loads((OUT/'ALL_INPUTS_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_input_freezer']
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    print('Native inputs/head hashes verified; opening target GT for explicit oracle construction.',flush=True)
    y=np.load(DATA/'target_labels.npy');sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    source_ids=np.flatnonzero(train)
    rows,metrics,classrows,audits=[],[],[],{}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}'
        source=np.load(folder/'source_rawZ.npy').astype(np.float64)[train]
        target=np.load(folder/'target_rawZ.npy').astype(np.float64)
        head=np.load(folder/'frozen_head.npz');w=head['weight'].astype(np.float64);b=head['bias'].astype(np.float64)
        native=np.load(folder/'target_predictions.npy');subset=masks(y,native)
        source_centers=np.stack([source[sy[train]==c].mean(0) for c in range(7)])
        target_centers=np.stack([target[y==c].mean(0) for c in range(7)])
        global_a,global_s,global_t,global_audit=affine(source,target)
        mapped={'G0':target.copy(),'G1':(target-global_t)@global_a+global_s,
                'G2':target+source_centers[y]-target_centers[y],'G3':np.empty_like(target)}
        matrices=[];conditional_audits={}
        for c in range(7):
            mask=y==c;a,ms,mt,check=affine(source[sy[train]==c],target[mask])
            mapped['G3'][mask]=(target[mask]-mt)@a+ms
            matrices.append(a);conditional_audits[str(c+1)]=check
        np.savez_compressed(folder/'oracle_transform_parameters.npz',global_affine=global_a,
            global_source_mean=global_s,global_target_mean=global_t,class_source_means=source_centers,
            class_target_means=target_centers,class_affine_matrices=np.stack(matrices))
        audits[str(seed)]={'G1':global_audit,'G3':conditional_audits,'target_GT_used_for_G2_G3_routing_and_statistics':True}
        outputs={}
        for group in GROUPS:
            z=mapped[group];assert np.isfinite(z).all() and (np.linalg.norm(z,axis=1)>1e-12).all()
            np.save(folder/f'{group}_rawZ.npy',z)
            ids,cos,ties=source_knn(z,source,source_ids)
            logits=z@w.T+b;pred=logits.argmax(1)
            if group=='G0':assert np.array_equal(pred,native)
            data={'source_indices':ids,'source_cosines':cos,'frozen_classifier_logits':logits,
                  'frozen_classifier_predictions':pred}
            for k in KS:
                counts=np.stack([(sy[ids[:,:k]]==c).sum(1) for c in range(7)],axis=1)
                data[f'source_counts_k{k}']=counts;data[f'source_knn_predictions_k{k}']=counts.argmax(1)
            np.savez_compressed(folder/f'{group}_oracle_readouts.npz',**data);outputs[group]=data
            norm=np.linalg.norm(z,axis=1)
            audits[str(seed)][group+'_feature_audit']={'source_cutoff_tie_rows':ties,
                'norm_min':float(norm.min()),'norm_mean':float(norm.mean()),'norm_max':float(norm.max())}
        for group in GROUPS:
            out=outputs[group];clf=out['frozen_classifier_predictions']
            clfmetric=metric(y,clf)
            for k in KS:
                knn=out[f'source_knn_predictions_k{k}'];knnmetric=metric(y,knn)
                support,smargin=margin(out[f'source_counts_k{k}'],y,k)
                bsupport,bmargin=margin(outputs['G0'][f'source_counts_k{k}'],y,k)
                metrics.append({'seed':seed,'group':group,'k_source':k,
                    'source_knn_OA':knnmetric['OA_percent'],'source_knn_AA':knnmetric['AA_percent'],
                    'frozen_classifier_OA':clfmetric['OA_percent'],'frozen_classifier_AA':clfmetric['AA_percent'],
                    'source_correct_support_micro':float(support.mean()),'source_support_margin_micro':float(smargin.mean()),
                    'source_correct_support_macro':float(np.mean([support[y==c].mean() for c in range(7)])),
                    'source_support_margin_macro':float(np.mean([smargin[y==c].mean() for c in range(7)]))})
                np.savez_compressed(folder/f'{group}_k{k}_offline_signals.npz',target_labels=y,
                    source_correct_support=support,source_support_margin=smargin,
                    classifier_correct=clf==y,source_knn_correct=knn==y)
                for name,mask in subset.items():
                    n=int(mask.sum())
                    if not n:continue
                    restored=int((mask&(bmargin<=0)&(smargin>0)).sum());lost=int((mask&(bmargin>0)&(smargin<=0)).sum())
                    r={'seed':seed,'group':group,'k_source':k,'subset':name,'n':n,
                        'source_support_before':float(bsupport[mask].mean()),'source_support_after':float(support[mask].mean()),
                        'source_support_delta':float((support[mask]-bsupport[mask]).mean()),
                        'source_margin_before':float(bmargin[mask].mean()),'source_margin_after':float(smargin[mask].mean()),
                        'source_margin_delta':float((smargin[mask]-bmargin[mask]).mean()),
                        'source_strict_support_restored_n':restored,'source_strict_support_restored_fraction':float(restored/n),
                        'source_strict_support_lost_n':lost,'source_strict_support_lost_fraction':float(lost/n),
                        **classification_change(native,clf,y,mask,'classifier'),
                        **classification_change(outputs['G0'][f'source_knn_predictions_k{k}'],knn,y,mask,'source_knn')}
                    rows.append(r)
                    if name.startswith('C') and '_to_' not in name:classrows.append(r)
            print(seed,group,'oracle classifier OA/AA',clfmetric['OA_percent'],clfmetric['AA_percent'],flush=True)
        dump(folder/'oracle_audit.json',audits[str(seed)])
    writecsv('per_seed_metrics.csv',metrics);writecsv('subset_statistics.csv',rows)
    writecsv('per_class.csv',classrows);writecsv('confusion_pairs.csv',[r for r in rows if '_to_' in r['subset']])
    # Same-seed differences relative to identity for every endpoint.
    paired=[]
    for r in metrics:
        base=next(v for v in metrics if v['seed']==r['seed'] and v['group']=='G0' and v['k_source']==r['k_source'])
        paired.append({'seed':r['seed'],'group':r['group'],'k_source':r['k_source'],
                      **{'delta_'+k:r[k]-base[k] for k in r if k not in ['seed','group','k_source']}})
    writecsv('paired_deltas_vs_G0.csv',paired)
    aggregate=[]
    for group in GROUPS:
        for k in KS:
            rr=[r for r in metrics if r['group']==group and r['k_source']==k]
            for key in rr[0]:
                if key in ['seed','group','k_source']:continue
                values=[r[key] for r in rr]
                aggregate.append({'group':group,'k_source':k,'metric':key,'mean':float(np.mean(values)),
                                  'sample_std':float(np.std(values,ddof=1)),'values':values})
    writecsv('seed_aggregate.csv',aggregate)
    limitations=['Oracle true-class routing and same-evaluation-set moment estimation are optimistic and label-assisted.',
        'G1/G3 are the same fixed affine moment family; G2 is translation only. No map/ridge/strength selection.',
        'Poor G1 can reflect class-prior composition, not absence of every possible global transform.',
        'G2/G3 gains do not prove identifiable/deployable class-conditional adaptation or its causal source.',
        'Remaining errors under this affine family are compatible with local/nonlinear shifts but do not prove them.',
        'Class covariances may be singular; fixed regularization and residual/rank/norm audits are supplied.',
        'Native classifier gets raw transformed CLS; kNN uses normalized transformed directions, references unchanged.',
        'One scene pair, three optimization seeds; targetC4 only22 samples. No significance/generalization claim.']
    dump(OUT/'summary.json',{'metrics':metrics,'subsets':rows,'aggregate':aggregate,'paired_deltas':paired,
         'transform_audits':audits,'limitations':limitations,'seconds':time.monotonic()-start})
    # This oracle freeze is AFTER explicit GT-assisted construction, unlike native input freeze.
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ORACLE_RESULTS_FROZEN.json',{'files_sha256':artifacts,'target_GT_used_for_oracle_G2_G3':True,
         'native_inputs_frozen_before_target_GT':True,'no_training_backward_tuning_followup':True,
         'helper_code_sha256':sha(helper_path),'fixture_checks_passed':True})
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    print('All oracle maps/readouts saved and hashed; native inputs/checkpoints unchanged.',flush=True)

if __name__=='__main__':main()
