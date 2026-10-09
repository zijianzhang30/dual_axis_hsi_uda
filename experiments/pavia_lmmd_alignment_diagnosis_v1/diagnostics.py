"""Offline statistics and frozen-checkpoint gradients. No optimizer or parameter steps."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans

BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_lmmd_tokenizer_v1')
OUT=BASE.parent/'pavia_lmmd_alignment_diagnosis_v1'
spec=importlib.util.spec_from_file_location('immutable_lmmd_control',BASE/'common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
SEEDS=c.SEEDS
PHASES=(1,10,50,100)
GRADIENT_BATCHES=8
TARGET_N=8192
dump,sha=c.dump,c.sha

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x

def write(path,obj):dump(path,clean(obj))

def verify_files():
    for p,h in json.loads((OUT/'provenance.json').read_text())['sha256'].items():assert sha(p)==h,p

def rng_audit(seed):
    model,_=c.build(seed,'C');model=model.cuda().train()
    cpu=torch.get_rng_state().numpy().copy();initial=torch.cuda.get_rng_state().numpy().copy();states=[initial.copy()]
    x=torch.zeros(32,1,102,11,11,device='cuda')
    with torch.no_grad():
        for _ in range(3):model(x,x);states.append(torch.cuda.get_rng_state().numpy().copy())
    assert len(initial)==16
    a=[np.frombuffer(s.tobytes(),dtype='<u8') for s in states]
    delta=int(a[1][1]-a[0][1]);assert delta==176
    assert all(int(a[i+1][1]-a[i][1])==delta for i in range(3))
    assert np.array_equal(torch.get_rng_state().numpy(),cpu)
    history=[json.loads(r) for r in (BASE/('C_LMMD_%d'%seed)/'history.jsonl').read_text().splitlines()]
    for epoch,row in enumerate(history,1):
        h=hashlib.sha256()
        for step in range(38):
            state=np.array([int(a[0][0]),int(a[0][1])+delta*3*((epoch-1)*38+step)],dtype='<u8')
            h.update(cpu.tobytes());h.update(state.tobytes())
        assert h.hexdigest()==row['model_rng_sha256'],(seed,epoch)
    info={'seed':seed,'cuda_seed':int(a[0][0]),'initial_offset':int(a[0][1]),
          'offset_per_paired_forward':delta,'all100_historical_model_RNG_hashes':'exactly reconstructed'}
    del model,x;gc.collect();torch.cuda.empty_cache()
    return cpu,info

def set_query_rng(cpu,info,epoch,step):
    torch.set_rng_state(torch.from_numpy(cpu.copy()))
    offset=info['initial_offset']+info['offset_per_paired_forward']*3*((epoch-1)*38+step-1)
    raw=np.array([info['cuda_seed'],offset],dtype='<u8').view(np.uint8)
    torch.cuda.set_rng_state(torch.from_numpy(raw.copy()))
    return offset

def probabilities(logits):
    a=np.asarray(logits,dtype=np.float64);a=a-a.max(1,keepdims=True);a=np.exp(a)
    return a/a.sum(1,keepdims=True)

def ece(p,truth):
    p=np.asarray(p);truth=np.asarray(truth);total=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if mask.any():total+=mask.mean()*abs(p[mask].mean()-truth[mask].mean())
    return float(total)

def reliability(p,y):
    p=np.asarray(p);y=np.asarray(y);pred=p.argmax(1);conf=p.max(1);correct=pred==y
    cm=np.zeros((7,7),dtype=np.int64);np.add.at(cm,(y,pred),1)
    rows=[]
    for k in range(7):
        actual=y==k;assigned=pred==k;wrong=actual&~correct
        mass=p[:,k].sum()
        rows.append({'class':k+1,'n_GT':int(actual.sum()),'n_predicted':int(assigned.sum()),
            'recall_percent':100*correct[actual].mean() if actual.any() else np.nan,
            'pseudo_precision_percent':100*correct[assigned].mean() if assigned.any() else np.nan,
            'GT_mean_correct_class_probability':p[actual,k].mean() if actual.any() else np.nan,
            'LMMD_soft_membership_purity':p[actual,k].sum()/mass if mass>0 else np.nan,
            'classwise_binary_ECE':ece(p[:,k],actual),
            'wrong_prediction_mean_confidence':conf[wrong].mean() if wrong.any() else np.nan,
            'wrong_high_confidence_ge_09_fraction':(conf[wrong]>=.9).mean() if wrong.any() else np.nan,
            'confusion_row':cm[k].tolist()})
    return {'n':len(y),'OA':100*correct.mean(),'AA':float(np.mean([r['recall_percent'] for r in rows])),
            'ECE10':ece(conf,correct),'NLL':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
            'Brier_sum_over_classes':float(((p-np.eye(7)[y])**2).sum(1).mean()),'classes':rows,
            'prediction_counts':cm.sum(0).tolist(),'confusion_matrix':cm.tolist()}

def distances(x,centers):
    return np.maximum((x*x).sum(1,keepdims=True)+(centers*centers).sum(1)[None,:]-2*x@centers.T,0)

def moments(x):
    centered=x-x.mean(0);cov=centered.T@centered/max(len(x)-1,1)
    eigen=np.maximum(np.linalg.eigvalsh(cov),0);trace=eigen.sum();q=eigen[eigen>0]/trace if trace>0 else np.array([])
    return {'covariance_trace':float(trace),'effective_rank':float(np.exp(-(q*np.log(q)).sum())) if len(q) else 0.,
            'mean_feature_norm':float(np.linalg.norm(x,axis=1).mean())}

def geometry(sf,tf,sy,ty,p):
    sf=sf.astype(np.float64);tf=tf.astype(np.float64)
    sc=np.stack([sf[sy==k].mean(0) for k in range(7)]);tc=np.stack([tf[ty==k].mean(0) for k in range(7)])
    cross=distances(tc,sc);cos=tc@sc.T/(np.linalg.norm(tc,axis=1)[:,None]*np.linalg.norm(sc,axis=1)[None,:]+1e-12)
    td=distances(tf,sc);sd=distances(sf,sc);pred=p.argmax(1);rows=[]
    for k in range(7):
        mask=ty==k;other=[j for j in range(7) if j!=k]
        margin=td[mask][:,other].min(1)-td[mask,k];sm=sy==k
        smargin=sd[sm][:,other].min(1)-sd[sm,k]
        weighted=(p[:,k,None]*tf).sum(0)/p[:,k].sum()
        wrong=mask&(pred!=k);right=mask&(pred==k)
        rows.append({'class':k+1,'source_n':int(sm.sum()),'target_n':int(mask.sum()),
            'source_target_same_class_centroid_squared_L2':cross[k,k],
            'source_target_same_class_centroid_cosine':cos[k,k],
            'target_centroid_nearest_source_class':int(cross[k].argmin())+1,
            'target_centroid_correct_vs_wrong_margin':cross[k,other].min()-cross[k,k],
            'target_sample_correct_source_margin_mean':margin.mean(),
            'target_sample_correct_source_margin_p10':np.quantile(margin,.1),
            'target_sample_correct_source_margin_median':np.median(margin),
            'target_sample_negative_correct_source_margin_fraction':(margin<0).mean(),
            'source_sample_correct_source_margin_mean':smargin.mean(),
            'target_nearest_source_centroid_accuracy_percent':100*(td[mask].argmin(1)==k).mean(),
            'wrong_predicted_subgroup_nearest_source_counts':np.bincount(td[wrong].argmin(1),minlength=7).tolist(),
            'wrong_predicted_subgroup_correct_source_margin_mean':(td[wrong][:,other].min(1)-td[wrong,k]).mean() if wrong.any() else np.nan,
            'correct_predicted_subgroup_correct_source_margin_mean':(td[right][:,other].min(1)-td[right,k]).mean() if right.any() else np.nan,
            'soft_target_centroid_to_source_squared_L2':float(((weighted-sc[k])**2).sum()),
            'soft_target_centroid_to_GT_target_squared_L2':float(((weighted-tc[k])**2).sum()),
            'source_moments':moments(sf[sm]),'target_moments':moments(tf[mask])})
    return {'classes':clean(rows),'cross_squared_L2':cross.tolist(),'cross_cosine':cos.tolist()}

def extract(model,dataset):
    loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False,drop_last=False,num_workers=0,
                                     generator=torch.Generator().manual_seed(731910))
    features=[];logits=[];model.eval()
    with torch.inference_mode():
        for x,_ in loader:
            n=len(x)
            if n<32:x=torch.cat((x,x[-1:].expand(32-n,-1,-1,-1,-1)),0)
            r=model(x.cuda(),x.cuda());features.append(r['ft'][:n].cpu().numpy());logits.append(r['lt'][:n].cpu().numpy())
    return np.concatenate(features),np.concatenate(logits)

def class_terms(mmd,main,sy,p):
    # Class decomposition follows exact native soft normalization + hard-presence gate.
    s=sy.detach().cpu().numpy();prob=p.detach().cpu().numpy();hard=prob.argmax(1)
    active=sorted(set(s)&set(hard));sv=np.eye(7)[s];ss=sv.sum(0);ss[ss==0]=100;sv=sv/ss
    ts=prob.sum(0);ts[ts==0]=100;tv=prob/ts
    kernel=mmd.guassian_kernel(main['fs'],main['ft']);terms={}
    for k in active:
        a=sv[:,k:k+1];b=tv[:,k:k+1]
        matrices=[np.asarray(v/len(active),dtype=np.float32) for v in (a@a.T,b@b.T,a@b.T)]
        w=[torch.from_numpy(v).cuda() for v in matrices]
        terms[k]=torch.sum(w[0]*kernel[:32,:32]+w[1]*kernel[32:,32:]-2*w[2]*kernel[:32,32:])
    return terms,active,tv

def flat_grad(loss,params):
    g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(v) if a is None else a).flatten() for v,a in zip(params,g)]).detach()

def comparison(a,b,scale=1.):
    na=float(torch.linalg.vector_norm(a));nb=float(torch.linalg.vector_norm(b));dot=float(torch.dot(a,b))
    return {'CE_grad_norm':na,'other_raw_grad_norm':nb,
            'cosine':dot/(na*nb) if na>1e-12 and nb>1e-12 else np.nan,
            'weighted_norm_ratio':scale*nb/na if na>1e-12 else np.nan,
            'CE_first_order_combined_descent_ratio':1+scale*dot/(na*na) if na>1e-12 else np.nan}

def parameter_slices(model):
    params=[];groups={};start=0
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        group=('CNN3D' if name.startswith('backbone.conv3d_features') else
               'CNN2D' if name.startswith('backbone.conv2d_features') else
               'tokenizer' if name.startswith('backbone.relation_scores') else
               'Transformer' if name.startswith('backbone.blocks') else
               'classifier' if name.startswith('backbone.nn1') else 'CLS_norm_other')
        groups.setdefault(group,[]).append((start,start+p.numel()));start+=p.numel();params.append(p)
    return params,groups

def modes(features,truth):
    labels=np.full(len(truth),-1,dtype=np.int64)
    for k in range(7):
        mask=truth==k
        # Offline feature statistics only; no neural network or classifier training.
        labels[mask]=KMeans(n_clusters=2,n_init=10,random_state=731911+k).fit_predict(features[mask])
    return labels

def mode_statistics(features,p,truth,labels):
    pred=p.argmax(1);rows=[]
    for k in range(7):
        mask=truth==k;centers=[]
        for j in range(2):
            chosen=mask&(labels==j);centers.append(features[chosen].mean(0))
            rows.append({'class':k+1,'mode':j,'n':int(chosen.sum()),
                'recall_percent':float(100*(pred[chosen]==k).mean()),**moments(features[chosen].astype(np.float64))})
        sep=float(((centers[0]-centers[1])**2).sum())
        for r in rows[-2:]:r['between_fixed_mode_squared_L2']=sep
    return rows
