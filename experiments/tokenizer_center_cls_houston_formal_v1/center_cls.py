"""Minimal center-conditioned CLS; all native downstream operations retained."""
import inspect,textwrap
from types import MethodType
import torch
from torch import nn

def center_delta(self,index):
 v=self._cq_centers[index]
 return self.center_projection(v if self._cq_arm=='CQ1' else torch.zeros_like(v)).unsqueeze(1)

def install(model,arm,seed):
 assert arm in ('CQ0','CQ1') and len(model.blocks)==3
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+51000).get_state())
  projection=nn.Linear(64,64,bias=False)
  nn.init.zeros_(projection.weight)
 model.center_projection=projection;model._cq_arm=arm;model._cq_centers=[]
 original_token=model._forward_semantic_tokens
 def token(features):
  assert features.shape[1:]==(64,13,13)
  model._cq_centers.append(features[:,:,6,6])
  return original_token(features)
 model._forward_semantic_tokens=token
 model._cq_delta=MethodType(center_delta,model)
 native=model._original_forward_func
 source=textwrap.dedent(inspect.getsource(native))
 a='x = torch.cat((cls_tokens, T), dim=1)';b='x_tar = torch.cat((cls_tokens, T_tar), dim=1)'
 assert source.count(a)==source.count(b)==1
 source=source.replace(a,'x = torch.cat((cls_tokens + self._cq_delta(0), T), dim=1)').replace(b,'x_tar = torch.cat((cls_tokens + self._cq_delta(1), T_tar), dim=1)')
 namespace=dict(native.__globals__);exec(compile(source,'<center_cls_native_forward>','exec'),namespace)
 model._original_forward_func=namespace[native.__name__]
 model._cq_native_forward_source=source
 original_forward=model.forward.__func__
 def forward(self,*args,**kwargs):
  self._cq_centers=[]
  result=original_forward(self,*args,**kwargs)
  assert len(self._cq_centers)==2
  return result
 model.forward=MethodType(forward,model)
