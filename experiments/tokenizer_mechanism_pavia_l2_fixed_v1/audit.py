"""Fail closed before training; local RNG only, true MAT and both memory orders."""
import pathlib,json,inspect,hashlib
import numpy as np
from scipy.io import loadmat
from common import OUT,environment,sha,dump
DATA=pathlib.Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia')
def cubehash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def measure(x):
 flat=x.reshape(-1,102);indices=np.flatnonzero(np.any(flat!=0,axis=1));assert len(indices)>=4096
 chosen=np.random.default_rng(41000).choice(indices,4096,replace=False)
 n=np.linalg.norm(flat[chosen].astype(np.float64),axis=1)
 maxerr=0.
 for start in range(0,len(indices),32768):
  v=flat[indices[start:start+32768]].astype(np.float64);maxerr=max(maxerr,float(np.max(abs(np.linalg.norm(v,axis=1)-1))))
 return {'sample_count':4096,'min':float(n.min()),'mean':float(n.mean()),'max':float(n.max()),'std':float(n.std()),'all_nonzero_max_error':maxerr,'passed':maxerr<1e-5 and abs(float(n.mean())-1)<1e-5}
def run(save=True):
 loader=environment()[7];report={'passed':True,'threshold':1e-5,'normalized_cube_sha256':{},'scenes':{},'load_scene':inspect.getfile(loader.load_scene)}
 for name,domain in [('paviaU','source'),('pavia','target')]:
  actual,gt=loader.load_scene(DATA,name);raw=loadmat(DATA/(name+'.mat'))[name].astype(np.float32)
  entry={'real_mat':measure(actual),'raw_layout':{'C':bool(raw.flags.c_contiguous),'F':bool(raw.flags.f_contiguous)},'mat_sha256':sha(DATA/(name+'.mat')),'layouts':{}}
  # Execute the ACTUAL frozen load_scene function with a substituted MAT reader,
  # not a duplicated normalization formula. Restore even after failure.
  original=loader.loadmat
  try:
   for order in ['C','F']:
    def fake(path,order=order):
     if pathlib.Path(path).name==name+'.mat':return {name:np.array(raw,order=order,copy=True)}
     return original(path)
    loader.loadmat=fake;result,_=loader.load_scene(DATA,name)
    entry['layouts'][order]={**measure(result),'same_normalized_cube_hash':cubehash(result)==cubehash(actual)}
  finally:loader.loadmat=original
  report['normalized_cube_sha256'][domain]=cubehash(actual);report['scenes'][domain]=entry
  report['passed'] &= entry['real_mat']['passed'] and all(v['passed'] and v['same_normalized_cube_hash'] for v in entry['layouts'].values())
 report['passed']=bool(report['passed'])
 if save:dump(OUT/'normalization_audit.json',report)
 assert report['passed'],'NORMALIZATION AUDIT FAILED: no automatic patch or continuation'
 # Per-run computed cubes must match the locked preflight hashes.
 if not save:
  locked=json.loads((OUT/'normalization_audit.json').read_text())
  assert report['normalized_cube_sha256']==locked['normalized_cube_sha256']
  for domain in ['source','target']:assert report['scenes'][domain]['mat_sha256']==locked['scenes'][domain]['mat_sha256']
 return report
if __name__=='__main__':print(json.dumps(run(),indent=2))
