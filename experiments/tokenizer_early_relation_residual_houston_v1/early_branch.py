import torch
from torch import nn

def install(model,seed):
 assert not hasattr(model,'spectral_relation_branch')
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+71000).get_state())
  model.spectral_relation_branch=nn.Conv3d(1,8,(3,1,1),padding=(1,0,0),bias=False)
 model.spectral_relation_gamma=nn.Parameter(torch.zeros(()))
 model._branch_record=False
 def residual(conv,args,absolute):
  patch=args[0]
  assert patch.shape[1:]==(1,48,13,13)
  relation=model.relation_scores.relations(patch).unsqueeze(1)
  relative=model.spectral_relation_branch(relation)
  assert relative.shape==absolute.shape
  if model._branch_record:
   model._branch_diagnostic={'absolute_RMS':float(absolute.detach().square().mean().sqrt()),'relative_RMS':float(relative.detach().square().mean().sqrt()),'gamma':float(model.spectral_relation_gamma.detach()),'residual_RMS':float((model.spectral_relation_gamma.detach()*relative.detach()).square().mean().sqrt()),'relative_shape':list(relative.shape),'center':[6,6]}
  return absolute+model.spectral_relation_gamma*relative
 model._spectral_relation_hook=model.conv3d_features[0].register_forward_hook(residual)
