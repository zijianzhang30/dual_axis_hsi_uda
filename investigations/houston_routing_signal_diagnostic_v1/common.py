import hashlib
import json
from pathlib import Path
import numpy as np
PROJECT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=PROJECT/'results/houston_routing_signal_diagnostic_v1'
CACHE=PROJECT/'results/houston_local_support_diagnostic_v1'
GEOM=PROJECT/'results/houston_class_geometry_diagnostic_v1'
FACT=PROJECT/'results/houston_transport_factorization_diagnostic_v1'
DATA=PROJECT/'results/tokenizer_stem_value_diagnostic_v1/data'
BASE=PROJECT/'results/tokenizer_mechanism_v2'
SEEDS=(2100,2101,2102)
STAGES=('V','T','Z')
KS=(5,15)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
