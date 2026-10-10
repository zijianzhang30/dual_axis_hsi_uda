import hashlib
import json
from pathlib import Path
import numpy as np
PROJECT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=PROJECT/'results/houston_unsupervised_transport_feasibility_v1'
CACHE=PROJECT/'results/houston_oracle_transport_diagnostic_v1'
DATA=PROJECT/'results/tokenizer_stem_value_diagnostic_v1/data'
BASE=PROJECT/'results/tokenizer_mechanism_v2'
SEEDS=(2100,2101,2102)
STAGES=('Z',)
KS=(5,15)
GROUPS=('F0','F1','F2','F3','Oracle_G2')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def softmax(a):
    e=np.exp(a-a.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
