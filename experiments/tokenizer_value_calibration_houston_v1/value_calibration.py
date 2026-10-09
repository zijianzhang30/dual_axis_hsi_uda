from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F

class ValueCalibration(nn.Module):
 def __init__(self,arm):
  super().__init__();assert arm in ('V0','V1');self.arm=arm
  self.net=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,64))
  self.gamma=nn.Parameter(torch.zeros(()));self.record=False
 def normalized_relation(self,v):
  assert v.shape[1:]==(64,13,13)
  difference=v-v[:,:,6:7,6:7] if self.arm=='V1' else v
  return F.normalize(difference,p=2,dim=1,eps=1e-6)
 def forward(self,v):
  d=self.normalized_relation(v);update=self.net(d.permute(0,2,3,1)).permute(0,3,1,2)
  if self.record:
   self.diagnostic={'V_RMS':float(v.detach().square().mean().sqrt()),'update_RMS':float(update.detach().square().mean().sqrt()),'residual_RMS':float((self.gamma.detach()*update.detach()).square().mean().sqrt()),'gamma':float(self.gamma.detach()),'value_shape':list(v.shape),'center':[6,6],'full_V_gradient':True}
  return v+self.gamma*update

def calibrated_tokens(self,features):
 return self._value_original_tokenizer(self.value_calibration(features))

def install(model,seed,arm):
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+73000).get_state())
  model.value_calibration=ValueCalibration(arm)
 model._value_original_tokenizer=model._forward_semantic_tokens
 model._forward_semantic_tokens=MethodType(calibrated_tokens,model)
