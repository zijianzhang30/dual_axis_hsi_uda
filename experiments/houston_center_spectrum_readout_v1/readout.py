from types import MethodType
import hashlib
import torch
from torch import nn

def base_hash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():
  if n.startswith('center_readout.') or n=='center_projection':continue
  h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def forward(self,source,target,*args,**kwargs):
 assert self._readout_centers is None,'nested residual readout'
 assert source.shape[1:]==target.shape[1:]==(1,48,13,13)
 # The same tensors are then consumed by the unchanged normal main-forward path.
 self._readout_centers=(source[:,0,:,6,6],target[:,0,:,6,6]);self._readout_calls=0;self._readout_diagnostics={}
 try:
  out=self._readout_original_forward(source,target,*args,**kwargs)
  assert self._readout_calls==(4 if self.training else 1)
  return out
 finally:self._readout_centers=None

def install(m,arm,seed):
 assert arm in ('R1','R2') and not hasattr(m,'center_readout');before=base_hash(m);cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+116000).get_state());head=nn.Linear(48,7,bias=False);nn.init.zeros_(head.weight)
 m.center_readout=head
 if arm=='R2':
  gen=torch.Generator().manual_seed(seed+117000);a=torch.randn(64,48,generator=gen);q,_=torch.linalg.qr(a,mode='reduced');m.register_buffer('center_projection',q.T.contiguous());assert torch.allclose(m.center_projection@m.center_projection.T,torch.eye(48),atol=1e-6)
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all())) and base_hash(m)==before
 m._readout_arm=arm;m._readout_centers=None;m._readout_calls=0;m._readout_original_forward=m.forward;m.forward=MethodType(forward,m)
 def classifier_hook(mod,args,out):
  call=m._readout_calls;m._readout_calls+=1
  # Unused coupled outputs remain exactly the native implementation; no coupled loss.
  if m.training and call>=2:return out
  z=args[0];center=m._readout_centers[call if m.training else 1]
  feature=center if arm=='R1' else z@m.center_projection.T
  assert feature.shape==(len(z),48);residual=m.center_readout(feature)
  with torch.no_grad():
   m._readout_diagnostics['source' if m.training and call==0 else 'target']={'input_norm_mean':float(feature.detach().norm(dim=1).mean()),'Z_norm_mean':float(z.detach().norm(dim=1).mean()),'base_logit_rms':float(out.detach().square().mean().sqrt()),'residual_logit_rms':float(residual.detach().square().mean().sqrt())}
  return out+residual
 m._readout_hook=m.nn1.register_forward_hook(classifier_hook)
 return {'main_initial_hash':before,'added_trainable_parameters':sum(p.numel() for p in m.center_readout.parameters()),'projection_sha256':hashlib.sha256(m.center_projection.numpy().tobytes()).hexdigest() if arm=='R2' else None,'CPU_CUDA_RNG_preserved':True}
