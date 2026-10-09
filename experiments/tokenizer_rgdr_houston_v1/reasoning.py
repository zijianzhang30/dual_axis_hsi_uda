"""Detached source prototype memory, dense vs relation-only sparse attention."""
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F

class PrototypeReasoning(nn.Module):
 def __init__(self,arm):
  super().__init__();assert arm in ['P0','P1'];self.arm=arm;self.heads=8;self.top_m=8;self.ema=.9;self.residual=.1
  self.q=nn.Linear(64,64,bias=False);self.k=nn.Linear(64,64,bias=False);self.v=nn.Linear(64,64,bias=False);self.out=nn.Linear(64,64,bias=True);nn.init.zeros_(self.out.bias)
  self.register_buffer('M_T',torch.zeros(7,4,64));self.register_buffer('M_W',torch.zeros(7,4,169));self.register_buffer('valid',torch.zeros(7,dtype=torch.bool));self.register_buffer('updates',torch.zeros(7,dtype=torch.long));self.diagnostics=False;self.last=None
 @torch.no_grad()
 def update_source(self,t,w,labels):
  assert t.shape==(len(labels),4,64) and w.shape==(len(labels),4,169);assert labels.ndim==1 and ((labels>=0)&(labels<7)).all()
  for c in labels.unique():
   index=int(c);mask=labels==c;mt=t.detach()[mask].mean(0);mw=w.detach()[mask].mean(0)
   if self.valid[index]:self.M_T[index].mul_(.9).add_(mt,alpha=.1);self.M_W[index].mul_(.9).add_(mw,alpha=.1)
   else:self.M_T[index].copy_(mt);self.M_W[index].copy_(mw);self.valid[index]=True
   self.updates[index]+=1
 def forward(self,t,w):
  b,k,d=t.shape;assert (k,d)==(4,64) and w.shape==(b,4,169);valid=self.valid[:,None].expand(7,4).flatten();ids=torch.nonzero(valid).flatten()
  if not len(ids):return torch.zeros_like(t)
  # Selection has no gradient and has no class/prediction argument.
  with torch.no_grad():
   similarity=F.normalize(w.detach(),dim=-1,eps=1e-6)@F.normalize(self.M_W.flatten(0,1).detach(),dim=-1,eps=1e-6).T
   sparse_ids=similarity.masked_fill(~valid[None,None,:],float('-inf')).topk(min(self.top_m,len(ids)),dim=-1).indices
   selected=sparse_ids if self.arm=='P1' else ids[None,None,:].expand(b,k,-1)
  memory=self.M_T.flatten(0,1).detach()[selected];m=memory.shape[2]
  q=self.q(t).reshape(b,k,8,8);keys=self.k(memory).reshape(b,k,m,8,8);values=self.v(memory).reshape(b,k,m,8,8)
  logits=torch.einsum('bqhd,bqmhd->bqhm',q,keys)*(8**-.5);attention=logits.softmax(-1);result=torch.einsum('bqhm,bqmhd->bqhd',attention,values).reshape(b,k,d)
  if self.diagnostics:self.last={'selected':selected.detach(),'sparse_candidate_ids':sparse_ids.detach(),'similarity':similarity.detach(),'attention':attention.detach(),'valid':valid.detach()}
  return self.out(result)

def install(model,arm,seed):
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+61000).get_state());module=PrototypeReasoning(arm)
 model.prototype_reasoning=module;model._prototype_raw=[];model._prototype_last_w=None
 def weight_capture(m,inputs,output):model._prototype_last_w=output
 module._weight_hook=model.relation_scores.register_forward_hook(weight_capture)
 original_token=model._forward_semantic_tokens
 def token(features):
  raw=original_token(features);w=model._prototype_last_w
  if model.training:model._prototype_raw.append((raw.detach(),w.detach()))
  refined=raw+.1*model.prototype_reasoning(raw,w)
  if model.prototype_reasoning.diagnostics:model._prototype_capture={'raw':raw.detach(),'weights':w.detach(),'refined':refined.detach()}
  return refined
 model._forward_semantic_tokens=token
 original_forward=model.forward.__func__
 def forward(self,*args,**kwargs):
  self._prototype_raw=[]
  return original_forward(self,*args,**kwargs)
 model.forward=MethodType(forward,model)
 def update(self,source_labels):
  assert self.training and len(self._prototype_raw)==2,'Must update from paired training forward source record only'
  self.prototype_reasoning.update_source(*self._prototype_raw[0],source_labels);self._prototype_raw=[]
 model.update_source_memory=MethodType(update,model)
 def target_only(self,target,return_details=False):
  assert not self.training,'Target-only path is eval only'
  self._relation_pending=[target]
  try:
   tokens=self._original_tokenize_func(self,target);assert not self._relation_pending
   z=self.dropout(torch.cat((self.cls_token.expand(len(target),-1,-1),tokens),dim=1)+self.pos_embedding)
   for block in self.blocks:_,z,_,_=block(None,z,None,inference_target_only=True)
   z=self.norm(z)[:,0];logits=self.nn1(self.to_cls_token(z))
   return (logits,self._prototype_capture,self.prototype_reasoning.last) if return_details else logits
  finally:self._relation_pending=[]
 model.target_only=MethodType(target_only,model)
