"""Fixed-checkpoint, source-only one-vs-three prototype test. Workers never load target GT."""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_subprototype_diagnosis_v1')
spec=importlib.util.spec_from_file_location('frozen_c_lmmd',ROOT/'experiments/pavia_lmmd_tokenizer_v1/common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
PLAN={'seeds':[2100,2101,2102], 'epoch':100,'high_confidence':0.9,
 'prototype_counts':[1,3],'features':{'T':'actual tokenizer tokens [4,64] flattened, L2-normalized',
                                 'Z':'actual pre-classifier CLS [64], L2-normalized'},
 'source_split':'each actual seed-specific180/class; private RandomState(seed+110000), first120 fit, next60 calibrate',
 'fit':'source-only KMeans K=3, n_init20, random_state(seed+120000+stage*100+class); normalized fitted centers',
 'single':'normalized source-fit class mean; exactly the same120 fit samples as K=3',
 'distance':'nearest prototype cosine distance per class; all seven classes use the same count',
 'rules':{'support':'reject predicted-class distance exceeding its source-only calibration threshold',
          'relative_margin':'reject predicted-class distance minus nearest other-class distance exceeding its source-only threshold'},
 'threshold':'per-class rank58 of60 source calibration scores, nominal95% acceptance; fixed before target GT',
 'preprocessing':'unchanged P2 z-score+official ILDA,102 bands,11x11; eval running BN, original paired path, batch32,pad-and-trim',
 'no_target_selection':True,'no_training':True,'no_automatic_new_alignment':True,
 'limitations':['Only frozen epoch100 checkpoint geometry, not actual historical gradients.',
  'Source calibration is held out from reference construction, but seen during original network training; patches can overlap.',
  'Source calibration coverage is not guaranteed to transfer to target.',
  'Three KMeans subprototypes are descriptive source clusters, not established semantic subclasses.']}

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def prepare():
    assert not (OUT/'frozen_plan.json').exists(), 'Do not overwrite diagnostics'
    c.dump(OUT/'frozen_plan.json',PLAN)
    paths=[]
    for seed in PLAN['seeds']:
        f=c.run_folder('C_LMMD',seed)
        m=json.loads((f/'frozen_manifest.json').read_text());ck=next(v for v in m['checkpoints'] if v['epoch']==100)
        assert c.sha(ck['path'])==ck['sha256']
        paths.extend([Path(ck['path']),f/'frozen_manifest.json',f/'predictions_epoch100.npz',c.P2/'inputs'/('source_selection_%d.npz'%seed)])
    paths.extend([c.P2/'inputs/P2_source.npy',c.P2/'inputs/P2_target.npy',c.P2/'inputs/target_training_coords.npy',Path(__file__)])
    c.dump(OUT/'provenance.json',{str(p):c.sha(p) for p in paths})

@torch.no_grad()
def extract(model,dataset,formal=None):
    rows={'T':[],'Z':[]};logits=[];cache=[]
    original=model.backbone._tokenize
    def capture(self,x):
        result=original(x);cache.append(result);return result
    model.backbone._tokenize=MethodType(capture,model.backbone)
    error=0.;position=0
    try:
        for x,unused in torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0):
            n=len(x)
            if n<32:x=torch.cat([x,x[-1:].expand(32-n,*x.shape[1:])])
            cache.clear();r=model(x.cuda(),x.cuda());assert len(cache)==2
            t=cache[-1][:n];z=r['ft'][:n];l=r['lt'][:n].cpu().numpy()
            assert t.shape[1:]==(4,64) and z.shape[1:]==(64,)
            assert torch.isfinite(t).all() and torch.isfinite(z).all()
            rows['T'].append(t.flatten(1).cpu().numpy());rows['Z'].append(z.cpu().numpy());logits.append(l)
            if formal is not None:
                error=max(error,float(np.abs(l-formal[position:position+n]).max()));assert error<1e-6,error
            position+=n
    finally:model.backbone._tokenize=original
    return {k:unit(np.concatenate(v)) for k,v in rows.items()},np.concatenate(logits),error

def class_distances(x,prototypes):
    # [N,C,K] distances; minimum is taken only within each class.
    return np.maximum(1-np.einsum('nd,ckd->nck',x,prototypes),0).min(2)

def margins(d):
    return np.stack([d[:,k]-np.delete(d,k,axis=1).min(1) for k in range(7)],axis=1)

def worker(seed):
    assert json.loads((OUT/'frozen_plan.json').read_text())==PLAN
    torch.set_num_threads(2);start=time.time();dest=OUT/str(seed);dest.mkdir(exist_ok=False)
    original_load=np.load
    def checked(path,*args,**kwargs):
        assert 'evaluation_labels_OFFLINE_ONLY' not in str(path),'Worker must never load target GT'
        return original_load(path,*args,**kwargs)
    np.load=checked
    f=c.run_folder('C_LMMD',seed);model,_=c.build(seed,'C')
    state=torch.load(f/'checkpoints/epoch_100.pth',map_location='cpu',weights_only=False)
    assert state['epoch']==100;model.load_state_dict(state['model'],strict=True);model=model.cuda().eval()
    before=c.state_hash(model);b=c.base()
    selection=np.load(c.P2/'inputs'/('source_selection_%d.npz'%seed));y=selection['labels']
    fit=[];cal=[];rng=np.random.RandomState(seed+110000)
    for k in range(7):
        ids=np.flatnonzero(y==k);assert len(ids)==180;ids=rng.permutation(ids);fit.extend(ids[:120]);cal.extend(ids[120:])
    fit=np.array(fit);cal=np.array(cal)
    features,sl,_=extract(model,b.PatchSet('P2','source',selection['coords']))
    np.savez(dest/'source_features.npz',**features,logits=sl,labels=y,coords=selection['coords'],fit_indices=fit,calibration_indices=cal)
    prototypes={};thresholds={};calibration={};cluster_counts={}
    for stage,(name,x) in enumerate(features.items()):
        for count in (1,3):
            key=name+'_K%d'%count;p=[];counts=[]
            for k in range(7):
                samples=x[fit][y[fit]==k]
                if count==1:centers=samples.mean(0,keepdims=True);counts.append([len(samples)])
                else:
                    km=KMeans(n_clusters=3,n_init=20,random_state=seed+120000+stage*100+k).fit(samples)
                    centers=km.cluster_centers_;counts.append(np.bincount(km.labels_,minlength=3).tolist())
                p.append(unit(centers))
            prototypes[key]=np.stack(p);cluster_counts[key]=counts
            d=class_distances(x[cal],prototypes[key]);m=margins(d)
            thresholds[key+'_support']=np.array([np.sort(d[y[cal]==k,k])[57] for k in range(7)])
            thresholds[key+'_relative_margin']=np.array([np.sort(m[y[cal]==k,k])[57] for k in range(7)])
            calibration[key]={'nearest_class_accuracy':float((d.argmin(1)==y[cal]).mean()*100),
                'threshold_support':thresholds[key+'_support'].tolist(),
                'threshold_relative_margin':thresholds[key+'_relative_margin'].tolist(),
                'cluster_source_fit_counts':counts}
    np.savez(dest/'SOURCE_ONLY_REFERENCES.npz',**prototypes,**thresholds)
    c.dump(dest/'SOURCE_ONLY_CALIBRATION_LOCK.json',{'seed':seed,'source_metrics':calibration,
             'reference_sha256':c.sha(dest/'SOURCE_ONLY_REFERENCES.npz')})
    print('SOURCE LOCKED %d %s'%(seed,json.dumps(calibration)),flush=True)
    tc=np.load(c.P2/'inputs/target_training_coords.npy')
    formal=np.load(f/'predictions_epoch100.npz')['logits']
    tf,tl,error=extract(model,b.PatchSet('P2','target',tc),formal)
    scores={k:class_distances(tf[k.split('_')[0]],p) for k,p in prototypes.items()}
    np.savez(dest/'target_scores_NO_GT.npz',**scores)
    np.savez(dest/'target_features_NO_GT.npz',**tf)
    assert c.state_hash(model)==before and all(p.grad is None for p in model.parameters())
    c.dump(dest/'complete_NO_GT.json',{'seed':seed,'model_state_unchanged':True,'parameter_gradients_none':True,
       'formal_logits_max_error':error,'target_count':len(tc),'target_GT_loaded':False,
       'source_reference_locked_before_target_scores':True,'elapsed_seconds':time.time()-start})
    print('COMPLETE_NO_GT seed=%d seconds=%.1f'%(seed,time.time()-start),flush=True)

def percent(mask,val):return float(val[mask].mean()*100) if mask.any() else None

def summarize():
    for seed in PLAN['seeds']:assert (OUT/str(seed)/'complete_NO_GT.json').exists()
    c.dump(OUT/'ALL_REFERENCES_AND_SCORES_LOCKED_BEFORE_TARGET_GT.json',{
       str(seed):{name:c.sha(OUT/str(seed)/name) for name in ['SOURCE_ONLY_CALIBRATION_LOCK.json','SOURCE_ONLY_REFERENCES.npz','target_scores_NO_GT.npz']} for seed in PLAN['seeds']})
    truth=np.load(c.P2/'inputs/evaluation_labels_OFFLINE_ONLY.npy');rows=[];paired=[]
    for seed in PLAN['seeds']:
        scores=np.load(OUT/str(seed)/'target_scores_NO_GT.npz');refs=np.load(OUT/str(seed)/'SOURCE_ONLY_REFERENCES.npz')
        logits=np.load(c.run_folder('C_LMMD',seed)/'predictions_epoch100.npz')['logits']
        pro=torch.from_numpy(logits).softmax(1).numpy();pred=pro.argmax(1);high=pro.max(1)>=.9
        good3=(truth==2)&(pred==2)&high
        for stage in ('T','Z'):
            for count in (1,3):
                key=stage+'_K%d'%count;d=scores[key];relative=margins(d)
                for rule,v in [('support',d),('relative_margin',relative)]:
                    accepted=v<=refs[key+'_'+rule][None,:]
                    for klass in [3,6,None]:
                        bad=(pred==2)&high&(np.isin(truth,[3,6]) if klass is None else truth==klass)
                        correct=(pred==truth)&(np.isin(truth,[3,6]) if klass is None else truth==klass)
                        cohort=good3|bad;own=accepted[np.arange(len(truth)),truth]
                        rows.append({'seed':seed,'stage':stage,'prototype_count':count,'rule':rule,
                            'true_class':'C4+C7' if klass is None else 'C%d'%(klass+1),
                            'error_count':int(bad.sum()),'correct_C3_count':int(good3.sum()),
                            'wrong_C3_rejection_percent':percent(bad,~accepted[:,2]),
                            'correct_C3_false_rejection_percent':percent(good3,~accepted[:,2]),
                            'correct_C4_C7_count':int(correct.sum()),
                            'correct_C4_C7_own_class_retention_percent':percent(correct,own),
                            'reject_C3_and_accept_true_class_percent':percent(bad,(~accepted[:,2])&own),
                            'true_class_closer_than_C3_percent':percent(bad,d[np.arange(len(truth)),truth]<d[:,2]),
                            'nearest_prototype_class_is_true_percent':percent(bad,d.argmin(1)==truth),
                            'error_vs_correct_C3_AUROC':float(roc_auc_score(bad[cohort],v[cohort,2]))})
            one=scores[stage+'_K1'];multi=scores[stage+'_K3']
            for klass in (3,6):
                bad=(truth==klass)&(pred==2)&high;good=(truth==klass)&(pred==truth)
                paired.append({'seed':seed,'stage':stage,'true_class':'C%d'%(klass+1),'error_count':int(bad.sum()),
                   'single_true_closer_than_C3_percent':percent(bad,one[:,klass]<one[:,2]),
                   'multi_true_closer_than_C3_percent':percent(bad,multi[:,klass]<multi[:,2]),
                   'rescued_pairwise_percent':percent(bad,(one[:,klass]>=one[:,2])&(multi[:,klass]<multi[:,2])),
                   'damaged_pairwise_percent':percent(bad,(one[:,klass]<one[:,2])&(multi[:,klass]>=multi[:,2])),
                   'correct_class_nearest_retention_single_percent':percent(good,one.argmin(1)==truth),
                   'correct_class_nearest_retention_multi_percent':percent(good,multi.argmin(1)==truth)})
    for filename,data in [('offline_target_metrics.csv',rows),('paired_correspondence_changes.csv',paired)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    aggregates=[]
    keys=['wrong_C3_rejection_percent','correct_C3_false_rejection_percent','correct_C4_C7_own_class_retention_percent',
          'true_class_closer_than_C3_percent','nearest_prototype_class_is_true_percent','error_vs_correct_C3_AUROC']
    for stage in ('T','Z'):
      for count in (1,3):
       for rule in ('support','relative_margin'):
        for cls in ('C4','C7','C4+C7'):
            rr=[r for r in rows if (r['stage'],r['prototype_count'],r['rule'],r['true_class'])==(stage,count,rule,cls)]
            aggregates.append({'stage':stage,'prototype_count':count,'rule':rule,'true_class':cls,
                'statistics':{k:{'mean':float(np.mean([r[k] for r in rr])),
                  'sample_std':float(np.std([r[k] for r in rr],ddof=1)),
                  'per_seed':[r[k] for r in rr]} for k in keys}})
    original=json.loads((OUT/'provenance.json').read_text())
    assert all(c.sha(path)==h for path,h in original.items())
    c.dump(OUT/'summary.json',{'plan':PLAN,'rows':rows,'paired_correspondence':paired,'aggregates':aggregates,
       'original_files_unchanged':True,'no_training':True,'no_followups':True})
    for a in aggregates:
        if a['rule']=='relative_margin':print(json.dumps(a),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--seed',type=int,choices=[2100,2101,2102]);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.summarize:summarize()
    else:assert args.seed is not None;worker(args.seed)
