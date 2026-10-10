import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_error_atlas_v1';CACHE=ROOT/'results/houston_routing_signal_diagnostic_v1'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data';GEOM=ROOT/'results/houston_class_geometry_diagnostic_v1'
SEEDS=(2100,2101,2102);STAGES=('V','T','Z');DESCRIPTORS=(*STAGES,'S_RAW','R_FULL');RADIUS=6
PATTERNS={'000':'persistent-error','001':'Z-recovered_after_early-hard','010':'T-recovered_then_Z-damaged','011':'T-recovered_retained',
 '100':'V-evidence_lost_at_T','101':'Z-recovered_after_T-loss','110':'T-to-Z_damaged','111':'all-centroid-correct'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def write(name,rows):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def aggregate(rows,keys,metrics):
 groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 output=[]
 for kk,rr in groups.items():
  row=dict(zip(keys,kk))
  for key in metrics:
   v=np.array([r[key] for r in rr if r.get(key) is not None],float);row[key+'_mean']=float(v.mean()) if len(v) else None;row[key+'_std']=float(v.std(ddof=1)) if len(v)>1 else None
  output.append(row)
 return output
def stats(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];d={'n':len(x)}
 if not len(x):return {**d,**{k:None for k in ['mean','std','p05','p25','median','p75','p95']}}
 return {**d,'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else None,
  **dict(zip(['p05','p25','median','p75','p95'],map(float,np.quantile(x,[.05,.25,.5,.75,.95]))))}
