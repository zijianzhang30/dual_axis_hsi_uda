import json,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np,torch
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'houston_scene_relation_diagnostic_v1';BASE=B/'tokenizer_mechanism_v2';DATA=B/'tokenizer_stem_value_diagnostic_v1/data';SEEDS=(2100,2101,2102)
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def frozen():
 sys.path.insert(0,str(BASE));spec=importlib.util.spec_from_file_location('scene_frozen_C',BASE/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

def angles(cube):
 norm=np.linalg.norm(cube.astype(np.float64),axis=-1);valid=np.isfinite(cube).all(-1)&(norm>1e-12);x=np.asarray(cube,dtype=np.float64)/np.maximum(norm[...,None],1e-12)
 hh=np.arccos(np.clip(np.sum(x[:,:-1]*x[:,1:],-1),-1,1));vv=np.arccos(np.clip(np.sum(x[:-1]*x[1:],-1),-1,1));return valid,hh,vv

def graph(cube,tau):
 h,w,b=cube.shape;valid,hh,vv=angles(cube);idx=np.arange(h*w).reshape(h,w);neighbors=np.tile(idx.flatten()[:,None],(1,4));distance=np.zeros((h*w,4),np.float64);linked=np.zeros((h*w,4),bool)
 for k,a,j,d in [(0,idx[1:],idx[:-1],vv),(1,idx[:-1],idx[1:],vv),(2,idx[:,1:],idx[:,:-1],hh),(3,idx[:,:-1],idx[:,1:],hh)]:
  i=a.ravel();v=j.ravel();mask=valid.flatten()[i]&valid.flatten()[v];neighbors[i,k]=v;distance[i,k]=d.ravel();linked[i,k]=mask
 degree=linked.sum(1);uniform=np.divide(linked.astype(float),degree[:,None],out=np.zeros_like(distance),where=degree[:,None]>0);logw=-(distance/tau)**2;logw[~linked]=-np.inf;mx=logw.max(1);mx[degree==0]=0;raw=np.exp(logw-mx[:,None]);weighted=np.divide(raw,raw.sum(1,keepdims=True),out=np.zeros_like(raw),where=degree[:,None]>0)
 assert np.isfinite(weighted).all() and np.allclose(weighted.sum(1)[degree>0],1)
 return neighbors,uniform,weighted,degree,valid

def aggregate(p,neighbors,weights,alpha=.5):
 q=(p[neighbors]*weights[...,None]).sum(1);has=weights.sum(1)>0;out=p.copy();out[has]=(1-alpha)*p[has]+alpha*q[has];assert np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-7);return out

def verify():
 for p,h in json.loads((OUT/'input_and_code_manifest.json').read_text()).items():assert sha(p)==h,p

class AllPixels(torch.utils.data.Dataset):
 def __init__(self):self.cube=np.load(DATA/'target_cube.npy',mmap_mode='r');self.h,self.w=np.load(OUT/'physical_shape.npy');assert self.cube.shape==(self.h+12,self.w+12,48)
 def __len__(self):return int(self.h*self.w)
 def __getitem__(self,i):
  row,col=divmod(i,int(self.w));x=np.array(self.cube[row:row+13,col:col+13],np.float32);assert x.shape==(13,13,48)
  return torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0)
