import json,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np,torch
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'houston_coordinate_context_diagnostic_v1';DIAG=B/'tokenizer_stem_value_diagnostic_v1';BASE=B/'tokenizer_mechanism_v2';SEEDS=(2100,2101,2102);OFFSETS=np.array([(0,0),(2,0),(-2,0),(0,2),(0,-2)],dtype=np.int64);ANCHORS=6-OFFSETS
PROBE={'scaler':'StandardScaler,fit on source training five-view tokens only','C':1.,'solver':'lbfgs','max_iter':2000,'tol':1e-6,'random_state':98000,'class_weight':None,'fit_intercept':True,'hyperparameter_selection':'none,fixed historical probe config','input':'flattened K4xD64 tokens;one shared probe for all five contexts'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def frozen():
 sys.path.insert(0,str(BASE));spec=importlib.util.spec_from_file_location('context_frozen_C',BASE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c
def verify():
 for p,h in json.loads((OUT/'input_and_code_manifest.json').read_text()).items():assert sha(p)==h,p
class Windows(torch.utils.data.Dataset):
 def __init__(self,domain,offset):self.cube=np.load(DIAG/'data'/('%s_cube.npy'%domain),mmap_mode='r');self.coords=np.load(OUT/'data'/('%s_coords_padded.npy'%domain));self.offset=offset
 def __len__(self):return len(self.coords)
 def __getitem__(self,i):
  p=self.coords[i];center=p+self.offset;patch=np.array(self.cube[center[0]-6:center[0]+7,center[1]-6:center[1]+7],np.float32)
  assert patch.shape==(13,13,48);anchor=6-self.offset;assert np.array_equal(patch[anchor[0],anchor[1]],self.cube[p[0],p[1]])
  return torch.from_numpy(patch.transpose(2,0,1)).unsqueeze(0)
def tokens(model,x,anchor):
 assert not model.training
 f=model.conv3d_features(x);v=model.conv2d_features(f.reshape(len(x),-1,13,13));assert v.shape==(len(x),64,13,13)
 w=model.relation_scores(x,coordinates=anchor);assert w.shape==(len(x),4,169)
 return torch.einsum('bln,bcn->blc',w,v.flatten(2))
