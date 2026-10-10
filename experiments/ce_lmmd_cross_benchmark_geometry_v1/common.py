import importlib.util,hashlib,json
from pathlib import Path
import numpy as np,torch
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'ce_lmmd_cross_benchmark_geometry_v1';SEEDS=(2100,2101,2102);STAGES=('F3D','V','V_mean','T','Z')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def folder(dataset,objective,seed):
 if dataset=='houston':return B/('tokenizer_mechanism_v2/houston/C_%d'%seed) if objective=='CE' else B/('cross_benchmark_lmmd_v1/houston/C_LMMD_%d'%seed)
 return B/('pavia_backbone_adaptation_v1/O_CE_%d'%seed) if objective=='CE' else B/('pavia_lmmd_tokenizer_v1/C_LMMD_%d'%seed)
def verify():
 for p,h in json.loads((OUT/'input_and_code_hashes.json').read_text()).items():assert sha(p)==h,p

def environment(dataset):return module('geometry_frozen_'+dataset,B/('cross_benchmark_lmmd_v1/snapshot/common.py' if dataset=='houston' else 'pavia_lmmd_tokenizer_v1/common.py'))

class HoustonPatches(torch.utils.data.Dataset):
 def __init__(self,domain,coords):self.cube=np.load(B/('tokenizer_stem_value_diagnostic_v1/data/%s_cube.npy'%domain),mmap_mode='r');self.coords=coords
 def __len__(self):return len(self.coords)
 def __getitem__(self,i):
  row,col=self.coords[i];p=np.array(self.cube[row-6:row+7,col-6:col+7],np.float32);return torch.from_numpy(p.transpose(2,0,1)).unsqueeze(0),0
