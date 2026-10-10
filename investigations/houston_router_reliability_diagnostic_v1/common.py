import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_router_reliability_diagnostic_v1'
BASE=ROOT/'results/houston_routing_signal_diagnostic_v1'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS=(2100,2101,2102)
KS=(5,15)
NAMES=('R0','R1','R2','R3','R4','R5','R6')
STAGE={'R0':'Z','R1':'Z','R2':'Z','R3':'T','R4':'T','R5':'V','R6':'V'}
FAMILIES=(('R1','R3','R5'),('R2','R4','R6'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvwrite(name,rows):
    with (OUT/name).open('x',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def signals(a,router):
    """Prespecified orientations: high alternative confidence favors overriding R0."""
    result={}
    for name in a.files:
        if name.startswith(router+'__'):
            suffix=name.split('__',1)[1]
            if suffix!='predictions':result['alternative_'+suffix]=(a[name],1)
    for suffix in ['logit_margin','entropy','agreement_all','agreement_other_stages']:
        result['R0_'+suffix]=(a['R0__'+suffix],1 if suffix=='entropy' else -1)
    for family in ['centroid','knn']:
        result[family+'_all_stages_agree']=(a[family+'__all_stages_agree'],1)
    return result

def auc(y,score):
    from scipy.stats import rankdata
    y=np.asarray(y,bool);n=int(y.sum());m=len(y)-n
    if n==0 or m==0:return None
    ranks=rankdata(score,method='average')
    return float((ranks[y].sum()-n*(n+1)/2)/(n*m))
