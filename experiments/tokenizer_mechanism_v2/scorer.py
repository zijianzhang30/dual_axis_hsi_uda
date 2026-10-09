"""Only spatial scorer changes; CNN values and transformer are untouched."""
from types import MethodType
import torch
from torch import nn
EPS=1e-6

class RelationScores(nn.Module):
 def __init__(self,arm,bands=48,tokens=4):
  super().__init__();assert arm in 'BCDEF';self.arm=arm;self.bands=bands
  self.scorer=nn.Linear(bands,tokens) if arm in 'DE' else nn.Sequential(nn.Linear(bands,32),nn.ReLU(),nn.Linear(32,tokens))
 def anchor(self,x,coordinates=None):
  b,c,h,w=x.shape
  if coordinates is None:
   if h%2==0 or w%2==0:raise ValueError('Even patch requires actual augmented anchor coordinates')
   coordinates=x.new_tensor([h//2,w//2],dtype=torch.long).expand(b,2)
  coordinates=torch.as_tensor(coordinates,device=x.device,dtype=torch.long)
  if coordinates.ndim==1:coordinates=coordinates.expand(b,2)
  assert coordinates.shape==(b,2)
  assert ((coordinates[:,0]>=0)&(coordinates[:,0]<h)&(coordinates[:,1]>=0)&(coordinates[:,1]<w)).all()
  return x[torch.arange(b,device=x.device),:,coordinates[:,0],coordinates[:,1]].unsqueeze(-1).unsqueeze(-1)
 def relations(self,patch,coordinates=None,arm=None):
  assert patch.ndim==5 and patch.shape[1]==1 and patch.shape[2]==self.bands
  x=patch[:,0];arm=arm or self.arm
  anchor=x.mean((-2,-1),keepdim=True) if arm in 'BD' else self.anchor(x,coordinates)
  relation=x-anchor
  return relation if arm=='F' else relation/(x.std((-2,-1),correction=0,keepdim=True)+EPS)
 def forward(self,patch,coordinates=None,arm=None):
  u=self.relations(patch,coordinates,arm).permute(0,2,3,1)
  return self.scorer(u).permute(0,3,1,2).flatten(2).softmax(-1)

def semantic_tokens(self,features):
 assert self._relation_pending,'Tokenization must run through paired forward'
 patch=self._relation_pending.pop(0)
 assert features.shape[0]==patch.shape[0] and features.shape[-2:]==patch.shape[-2:]
 weights=self.relation_scores(patch)
 assert weights.shape[1]==self.token_len
 return torch.einsum('bln,bcn->blc',weights,features.flatten(2))

def forward(self,source,target,*args,**kwargs):
 assert not self._relation_pending,'Nested relation forward'
 self._relation_pending=[source,target]
 try:
  output=self._relation_original_forward(self,source,target,*args,**kwargs)
  assert not self._relation_pending
  return output
 finally:self._relation_pending=[]

def install(model,arm,seed,bands):
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+41000).get_state())
  scores=RelationScores(arm,bands,model.token_len)
 del model.conv_a;model.relation_scores=scores;model._relation_pending=[]
 model._relation_original_forward=model.forward.__func__
 model.forward=MethodType(forward,model);model._forward_semantic_tokens=MethodType(semantic_tokens,model)
