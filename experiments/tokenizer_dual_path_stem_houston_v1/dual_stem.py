import torch
from torch import nn

class DualPathStem(nn.Module):
 def __init__(self,original,arm):
  super().__init__();assert arm in ('N0','N1');self.arm=arm
  self.absolute=nn.Conv3d(1,4,3,padding=1,bias=True)
  self.relative=nn.Conv3d(1,4,3,padding=1,bias=True)
  with torch.no_grad():
   self.absolute.weight.copy_(original.weight[:4]);self.absolute.bias.copy_(original.bias[:4])
   self.relative.weight.copy_(original.weight[4:]);self.relative.bias.copy_(original.bias[4:])
  self.record=False
 def relation(self,patch):
  assert patch.shape[1:]==(1,48,13,13)
  center=patch[:,:,:,6:7,6:7]
  return (patch-center)/(patch.std((-2,-1),correction=0,keepdim=True)+1e-6)
 def forward(self,patch):
  r=self.relation(patch) if self.arm=='N1' else patch
  absolute=self.absolute(patch);relative=self.relative(r)
  assert absolute.shape==relative.shape
  if self.record:
   self.diagnostic={'absolute_RMS':float(absolute.detach().square().mean().sqrt()),'second_path_RMS':float(relative.detach().square().mean().sqrt()),'path_shapes':list(absolute.shape),'fusion':'concatenate channel dim; jointly projected by original Conv2D','center':[6,6]}
  return torch.cat((absolute,relative),dim=1)

def install(model,seed,arm):
 old=model.conv3d_features[0]
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+72000).get_state())
  model.conv3d_features[0]=DualPathStem(old,arm)
