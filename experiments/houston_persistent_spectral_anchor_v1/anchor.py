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
 if self._spectral_arm=='S1':x=torch.cat((x,a[:,None]),1);xt=torch.cat((xt,at[:,None]),1)
 x=self.dropout(x);xt=self.dropout(xt);infer=not self.training;xf=xt
 for l,blk in enumerate(self.blocks):
  if self._spectral_arm=='S2':
   def read(v,anchor):
    anchor=anchor[:,None].expand(-1,v.shape[1],-1)
    gate=torch.sigmoid(self.spectral_gates[l](torch.cat((v,anchor),-1)))
    return v+self.spectral_alpha[l]*gate*anchor
   if not infer:x=read(x,a);xf=read(xf,at)
   xt=read(xt,at)
  x,xt,xf,xfs=blk(x,xt,xf,inference_target_only=infer)
 if infer:
  xt=self.norm(xt);out=self.nn1(self.to_cls_token(xt[:,0]))
  return (None,out,None,xt[:,0]) if return_feat_prob else (None,out,None)
 x=self.norm(x);xt=self.norm(xt);xf=self.norm(xf);xfs=self.norm(xfs)
 return self.nn1(self.to_cls_token(x[:,0])),self.nn1(self.to_cls_token(xt[:,0])),self.nn1(self.to_cls_token(xf[:,0])),self.nn1(self.to_cls_token(xfs[:,0]))
def install(m,arm,seed=2100):
 before=base_hash(m);cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
 if arm!='S0':
  with torch.random.fork_rng(devices=[]):
   torch.set_rng_state(torch.Generator().manual_seed(seed+126000).get_state())
   encoder=nn.Sequential(nn.Linear(48,64),nn.LayerNorm(64))
   gates=nn.ModuleList([nn.Linear(128,1) for _ in range(3)]) if arm=='S2' else None
  m.spectral_encoder=encoder
  if arm=='S2':m.spectral_gates=gates;m.spectral_alpha=nn.Parameter(torch.zeros(3))
  m._spectral_arm=arm;m._original_forward_func=native_forward
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all())) and before==base_hash(m)
 return {'base_initial_hash':before,'new_parameters':sum(p.numel() for n,p in m.named_parameters() if n.startswith('spectral_')),'RNG_unchanged':True}
