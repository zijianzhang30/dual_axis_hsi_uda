import hashlib
import torch
from torch import nn

def base_hash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():
  if n.startswith('spectral_'):continue
  h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def native_forward(self,source,target,inference_target_only=False,return_feat_prob=False):
 T=self._tokenize(source);Tt=self._tokenize(target);cls=self.cls_token.expand(len(source),-1,-1)
 x=torch.cat((cls,T),1);xt=torch.cat((cls,Tt),1);x+=self.pos_embedding;xt+=self.pos_embedding
 a=self.spectral_encoder(source[:,0,:,6,6]);at=self.spectral_encoder(target[:,0,:,6,6])
 x=self.dropout(x);xt=self.dropout(xt);infer=not self.training;xf=xt
 for blk in self.blocks:
  x,xt,xf,xfs=blk(x,xt,xf,inference_target_only=infer)
 if not infer:x=inject_cls(self,x,a)
 xt=inject_cls(self,xt,at)
 if infer:
  xt=self.norm(xt);out=self.nn1(self.to_cls_token(xt[:,0]))
  return (None,out,None,xt[:,0]) if return_feat_prob else (None,out,None)
 x=self.norm(x);xt=self.norm(xt);xf=self.norm(xf);xfs=self.norm(xfs)
 return self.nn1(self.to_cls_token(x[:,0])),self.nn1(self.to_cls_token(xt[:,0])),self.nn1(self.to_cls_token(xf[:,0])),self.nn1(self.to_cls_token(xfs[:,0]))
def inject_cls(self,v,anchor):
 cls=v[:,0];gate=torch.sigmoid(self.spectral_gate(torch.cat((cls,anchor),-1)))
 updated=cls+self.spectral_alpha[0]*gate*anchor
 return torch.cat((updated[:,None],v[:,1:]),1)
def install(m,arm,seed=2100):
 assert arm in ('S0','S3')
 before=base_hash(m);cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
 if arm=='S3':
  with torch.random.fork_rng(devices=[]):
   torch.set_rng_state(torch.Generator().manual_seed(seed+126000).get_state())
   encoder=nn.Sequential(nn.Linear(48,64),nn.LayerNorm(64));gate=nn.Linear(128,1)
  m.spectral_encoder=encoder;m.spectral_gate=gate;m.spectral_alpha=nn.Parameter(torch.zeros(1))
  m._original_forward_func=native_forward
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all())) and before==base_hash(m)
 return {'base_initial_hash':before,'new_parameters':sum(p.numel() for n,p in m.named_parameters() if n.startswith('spectral_')),'RNG_unchanged':True}
