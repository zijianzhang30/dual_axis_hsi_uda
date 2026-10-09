import contextlib
import torch
from torch import nn

class Adapter(nn.Module):
 def __init__(self):
  super().__init__();self.down=nn.Linear(64,16,bias=False);self.act=nn.GELU();self.up=nn.Linear(16,64,bias=False);nn.init.zeros_(self.up.weight)
 def forward(self,x):return self.up(self.act(self.down(x)))
def reset(model,inputs):
 for block in model.blocks:block._adapter_calls=0

def parallel(block):
 def hook(module,inputs,output):
  if block.training:
   i=block._adapter_calls;block._adapter_calls+=1
   if i>=2:return output
   role='source' if i==0 else 'target'
  else:role=block._adapter_eval_domain
  a=block.adapter_shared if hasattr(block,'adapter_shared') else getattr(block,'adapter_'+role)
  return output+a(inputs[0])
 return hook

def install(model,arm,seed):
 original={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+73000).get_state())
  for block in model.blocks:
   if arm=='R1':block.adapter_shared=Adapter()
   elif arm=='R2':
    block.adapter_source=Adapter();block.adapter_target=Adapter();block.adapter_target.load_state_dict(block.adapter_source.state_dict())
   else:raise ValueError(arm)
   block._adapter_eval_domain='target';block._adapter_calls=0;block.mlp.register_forward_hook(parallel(block))
 model.register_forward_pre_hook(reset)
 assert torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
 assert sum(p.numel() for name,p in model.named_parameters() if '.adapter_' in name)==(6144 if arm=='R1' else 12288)

@contextlib.contextmanager
def eval_domain(model,role):
 assert role in ('source','target');old=[]
 for b in model.blocks:
  if hasattr(b,'_adapter_eval_domain'):old.append((b,b._adapter_eval_domain));b._adapter_eval_domain=role
 try:yield
 finally:
  for b,v in old:b._adapter_eval_domain=v

def gradients(model):return {k:0. if p.grad is None else float(p.grad.norm()) for k,p in model.named_parameters() if '.adapter_' in k}
