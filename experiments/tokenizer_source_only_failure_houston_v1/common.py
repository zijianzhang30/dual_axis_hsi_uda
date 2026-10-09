import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import torch

PROJECT=Path('/home/zhangzj26/dual_axis_hsi_uda')
BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_source_only_failure_houston_v1')
SEEDS=(2100,2101,2102)
ARMS=('A','C')

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def state_hash(model):
 h=hashlib.sha256()
 for value in model.state_dict().values():h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def environment():
 snapshot=OUT/'snapshot/BiDA'
 sys.path.insert(0,str(snapshot/'experiments/bida_memory_v1'));sys.path.insert(0,str(snapshot))
 from experiments.bida_memory_v1.run import make_loaders as houston_loaders
 from models.get_model import get_model
 from models.BiDA import BiDAnet
 from utils.scheduler import load_scheduler
 from utils.utils_HSI import seed_worker
 from utils.dataset import HSIDataset,sample_gt
 joint=module('frozen_joint_normalization',OUT/'snapshot/normalization.py')
 helper=module('frozen_metric_helpers',OUT/'snapshot/helpers.py')
 pavia=module('frozen_pavia_loaders',OUT/'snapshot/pavia_loaders.py')
 return houston_loaders,get_model,BiDAnet,load_scheduler,seed_worker,joint,helper,pavia

def source_loaders(seed,workers=4):
 env=environment()
 from utils.dataset import load_mat_hsi,HSIDataset,sample_gt
 from torch.utils.data import DataLoader
 image,gt,_=load_mat_hsi('Houston13','/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',norm='normband')
 train,val=sample_gt(gt,.95,seed,mode='random');pad=np.pad(image,((6,6),(6,6),(0,0)),mode='reflect')
 generator=torch.Generator().manual_seed(seed)
 datasets=[HSIDataset(pad,np.pad(g,((6,6),(6,6)),mode='reflect'),13,data_aug=aug) for g,aug in [(train,True),(val,False)]]
 loaders=[DataLoader(d,batch_size=128,num_workers=workers,drop_last=False,shuffle=(i==0),generator=generator) for i,d in enumerate(datasets)]
 return loaders,image

def build(seed,arm,dataset='houston'):
 from argparse import Namespace
 from types import MethodType
 import scorer
 env=environment();args=Namespace(model='BiDA',lr=.01,num_tokens=4,dim=64,depth=3,loss_type='softmax',labelsmooth='off')
 model=env[1]('BiDA','Houston13',13,args);unused=env[1]('BiDA','Houston13',13,args,ema=True);del unused
 original=state_hash(model)
 if arm=='C':scorer.install(model,'C',seed,48)
 assert model.conv3d_features[1].momentum==model.conv2d_features[1].momentum==.1
 def forward(self,x,*,return_features=False):
  assert x.ndim==5 and x.shape[1:]==(1,48,13,13)
  self._relation_pending=[x] if arm=='C' else []
  v=self.conv3d_features(x);b,c,bands,h,w=v.shape;v=self.conv2d_features(v.reshape(b,c*bands,h,w))
  tokens=self._forward_semantic_tokens(v)
  assert not self._relation_pending
  cls=self.cls_token.expand(b,-1,-1);z=self.dropout(torch.cat((cls,tokens),dim=1)+self.pos_embedding)
  for block in self.blocks:_,z,_,_=block(None,z,None,inference_target_only=True)
  z=self.norm(z)[:,0];logits=self.nn1(self.to_cls_token(z))
  if return_features:
   weights=self.relation_scores(x) if arm=='C' else self.conv_a(v).flatten(2).softmax(-1)
   return logits,{'V':v,'W':weights,'T':tokens,'Z':z}
  return logits
 model.forward=MethodType(forward,model)
 return model,original
