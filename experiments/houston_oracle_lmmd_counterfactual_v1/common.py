import json,hashlib,importlib.util,copy
from pathlib import Path
import numpy as np,torch
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'houston_oracle_lmmd_counterfactual_v1';SEEDS=(2100,2101,2102);PHASES=(10,50,100,200)
def module():
 spec=importlib.util.spec_from_file_location('oracle_immutable_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def verify():
 module().verify_files()
 for p,h in json.loads((OUT/'immutable_manifest.json').read_text()).items():assert sha(p)==h,p

class IndexedTarget(torch.utils.data.Dataset):
 def __init__(self,d):self.d=d
 def __len__(self):return len(self.d)
 def __getitem__(self,i):x,dummy=self.d[i];assert int(dummy)==0;return x,int(i)
def tensor_inputs(ds,ids):
 ds=copy.copy(ds)
 for key in ('data_aug','flip_augmentation','radiation_augmentation','mixture_augmentation'):setattr(ds,key,False)
 v=[ds[int(i)] for i in ids];return torch.stack([x for x,y in v]),torch.stack([y for x,y in v])
@torch.inference_mode()
def predict(m,x):
 m.eval();before={n:b.clone() for n,b in m.named_buffers()};out=[]
 for b in x.split(128):b=b.cuda();z=m(b,b)[1];assert torch.isfinite(z).all();out.append(z.cpu().numpy())
 assert all(torch.equal(v,dict(m.named_buffers())[n]) for n,v in before.items());return np.concatenate(out)
