import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_target_cluster_routing_diagnostic_v1'
CACHE=ROOT/'results/houston_routing_signal_diagnostic_v1'
FACT=ROOT/'results/houston_transport_factorization_diagnostic_v1'
LEARNED=ROOT/'results/houston_source_supervised_router_v1'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS=(2100,2101,2102)
STAGES=('T','Z')
CONFIG={'K':7,'init':'k-means++','n_init':10,'max_iter':300,'tol':.0001,
        'algorithm':'lloyd','random_state':0,'input':'float64 sample L2-normalized features',
        'source_centroid':'unit mean of unit source-training feature vectors',
        'matching':'Hungarian minimize 1-cosine; strict one-to-one',
        'Go_rule':'A stage is stable positive only if transport OA > D00 OA in all3seeds; if neither T nor Z qualifies: No-Go',
        'no_search_or_followup':True}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def csvwrite(name,rows):
    with (OUT/name).open('x',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def metric(y,p):
    cm=np.bincount(7*y+p,minlength=49).reshape(7,7);n=cm.sum(1)
    recall=np.divide(np.diag(cm),n,out=np.zeros(7,float),where=n>0)
    po=float(np.trace(cm)/len(y));pe=float(cm.sum(0)@cm.sum(1)/len(y)**2)
    return {'OA':100*po,'AA':100*float(recall[n>0].mean()),'Kappa':(po-pe)/(1-pe) if pe<1 else None,
        'recall':100*recall,'confusion':cm}

def subsets(y,r0):
    correct=r0==y;out={'all':np.ones(len(y),bool),'R0_correct':correct,'R0_error':~correct}
    for c in range(7):
        for name,m in [('all',np.ones(len(y),bool)),('R0_correct',correct),('R0_error',~correct)]:out[f'C{c+1}_{name}']=(y==c)&m
        for j in range(7):
            mask=(y==c)&(r0==j)
            if c!=j and mask.any():out[f'C{c+1}_to_C{j+1}']=mask
    return out

def aggregate(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for group,rs in groups.items():
        row=dict(zip(keys,group))
        for key in metrics:
            a=np.array([r[key] for r in rs if r.get(key) is not None],float)
            row[key+'_mean']=float(a.mean()) if len(a) else None
            row[key+'_sample_std']=float(a.std(ddof=1)) if len(a)>1 else None
            row[key+'_valid_seeds']=len(a)
        out.append(row)
    return out
