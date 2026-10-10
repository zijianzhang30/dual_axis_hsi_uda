import torch
from torch import nn

def build(seed=2100):
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+146000).get_state())
  expert=nn.Sequential(nn.LayerNorm(48),nn.Linear(48,64),nn.GELU(),nn.Linear(64,7))
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
 assert sum(p.numel() for p in expert.parameters())==3687
 return expert
