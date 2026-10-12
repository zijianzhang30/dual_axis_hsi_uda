import torch
from torch import nn

def build(bands,classes,seed):
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+146000).get_state());e=nn.Sequential(nn.LayerNorm(bands),nn.Linear(bands,64),nn.GELU(),nn.Linear(64,classes))
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()));assert sum(p.numel() for p in e.parameters())==66*bands+64+65*classes
 return e
class DropoutReplay:
 def __init__(self,calls):
  self.calls=calls;self.buffers={}
  for shape,p in calls:self.buffers[tuple(shape)]=torch.zeros(tuple(shape),device='cuda',dtype=torch.float32)
 @torch.no_grad()
 def advance(self):
  for shape,p in self.calls:torch.nn.functional.dropout(self.buffers[tuple(shape)],p=p,training=True)
