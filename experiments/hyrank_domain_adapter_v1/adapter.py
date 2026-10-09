"""Parallel deterministic FFN adapters. Never modify CNN, attention or tokenizer."""
import torch
from torch import nn

class Adapter(nn.Module):
 def __init__(self,width):
  super().__init__();self.down=nn.Linear(64,width,bias=False);self.act=nn.GELU();self.up=nn.Linear(width,64,bias=False)
  nn.init.zeros_(self.up.weight)
 def forward(self,x):return self.up(self.act(self.down(x)))

def reset(model,inputs):
 for b in model.blocks:b._adapter_calls=0

def add_parallel(block):
 def hook(mlp,inputs,output):
  if not block.training:domain='target'
  else:
   index=block._adapter_calls;block._adapter_calls+=1
   if index>=2:return output  # Historical unused coupled products never receive adapters.
   domain='source' if index==0 else 'target'
  a=block.adapter_shared if hasattr(block,'adapter_shared') else getattr(block,'adapter_'+domain)
  return output+a(inputs[0])
 return hook

def install(model,arm,seed):
 before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+73000).get_state())
  for block in model.blocks:
   if arm=='R1':block.adapter_shared=Adapter(32)
   elif arm=='R2':
    block.adapter_source=Adapter(16);block.adapter_target=Adapter(16)
    # Same starting low-rank transform for both domains, later trained separately.
    block.adapter_target.load_state_dict(block.adapter_source.state_dict())
   else:raise ValueError(arm)
   block._adapter_calls=0;block.mlp.register_forward_hook(add_parallel(block))
 model.register_forward_pre_hook(reset)
 assert torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 assert sum(p.numel() for name,p in model.named_parameters() if '.adapter_' in name)==12288

def gradient_stats(model):
 return {name:0. if p.grad is None else float(p.grad.norm()) for name,p in model.named_parameters() if '.adapter_' in name}
