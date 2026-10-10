import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_scene_shift_direction_diagnostic_v1'
CACHE=ROOT/'results/houston_transport_factorization_diagnostic_v1'
SEEDS=(2100,2101,2102)
EPS=(.02,.05,.10)
PROBES=('global_mean','covariance_rank4')
REPS=16
CONFIG={'feature':'native raw CLS Z64, no sample renormalization','eps':list(EPS),'rank':4,
        'random_repetitions':REPS,'random_seed':20261012,
        'mean':'unit(target all52901 mean - source all2520 mean)',
        'subspace':'top4 left singular vectors of separately centered unbiased covariance discrepancy Ct-Cs',
        'domain_attack':'exact worst true-vs-strongest-class margin within span at per-sample epsilon*rawZ norm',
        'fair_control':'same-rank isotropic random subspaces, same exact source-label-guided worst-margin attack',
        'clear_margin_rule':'domain mean margin drop > random-subspace adversarial repetition mean + 2 repetition SD, for all3model seeds at ALL3eps',
        'classification_rule':'domain flip rate > random-subspace adversarial repetition mean at eps.10 for all3model seeds',
        'no_target_GT_or_parameter_selection_or_followup':True}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def canonical(b):
    b=b.copy()
    for j in range(b.shape[1]):
        if b[np.argmax(abs(b[:,j])),j]<0:b[:,j]*=-1
    return b
def csvwrite(name,rows):
    with (OUT/name).open('x',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def true_margin(logits,y):
    other=logits.copy();other[np.arange(len(y)),y]=-np.inf
    return logits[np.arange(len(y)),y]-other.max(1)

def exact_attack(z,logits,w,y,basis,epsilon):
    """Closed-form exact min over competing affine margins and an L2 ball in span."""
    g=w[y,None,:]-w[None,:,:]
    coeff=g@basis;pn=np.linalg.norm(coeff,axis=2)
    pair=logits[np.arange(len(y)),y,None]-logits
    pair[np.arange(len(y)),y]=np.inf
    rho=epsilon*np.linalg.norm(z,axis=1)
    values=pair-rho[:,None]*pn;competitor=values.argmin(1)
    chosen=coeff[np.arange(len(y)),competitor];norm=np.linalg.norm(chosen,axis=1)
    direction=-chosen@basis.T/np.maximum(norm[:,None],1e-12)
    zero=norm<1e-12;direction[zero]=basis[:,0]
    delta=rho[:,None]*direction
    scores=logits+delta@w.T
    expected=values.min(1);actual=true_margin(scores,y)
    assert np.max(abs(expected-actual))<1e-9
    return delta,scores,competitor

def metric(logits,y,baseline):
    p=logits.argmax(1);bp=baseline.argmax(1);before=true_margin(baseline,y);after=true_margin(logits,y)
    recalls=[(p[y==c]==c).mean() for c in range(7) if (y==c).any()]
    return {'OA':float(100*(p==y).mean()),'AA':float(100*np.mean(recalls)),
        'margin_before_mean':float(before.mean()),'margin_after_mean':float(after.mean()),
        'margin_drop_mean':float((before-after).mean()),'margin_after_median':float(np.median(after)),
        'margin_after_q10':float(np.quantile(after,.1)),'flip_percent':float(100*(p!=bp).mean()),
        'baseline_OA':float(100*(bp==y).mean()),'n':len(y),
        'n_original_correct_damaged':int(((bp==y)&(p!=y)).sum()),
        'n_original_wrong_recovered':int(((bp!=y)&(p==y)).sum())}

def aggregate(rows,keys,values):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for group,rs in groups.items():
        out=dict(zip(keys,group))
        for v in values:
            a=np.array([r[v] for r in rs if r.get(v) is not None],float)
            out[v+'_mean']=float(a.mean()) if len(a) else None
            out[v+'_sample_std']=float(a.std(ddof=1)) if len(a)>1 else None
            out[v+'_n']=len(a)
        result.append(out)
    return result
