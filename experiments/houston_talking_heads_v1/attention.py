import types
import torch
import torch.nn.functional as F
from torch import nn
import study as s

def base_hash(m):
 h=__import__('hashlib').sha256()
 for n,v in m.state_dict().items():
  if 'talk_pre' in n or 'talk_post' in n:continue
  h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def mix(self,x,which):
 p=getattr(self,which)
 if self.talking_arm=='TH1':return x*p.reshape(1,self.num_heads,1,1)
 # Restore native B,H,N,N memory order so dropout maps the same RNG stream
 # to the same positions at identity initialization.
 return F.linear(x.permute(0,2,3,1),p,bias=None).permute(0,3,1,2).contiguous()

def self_weights(self,q,k):
 logits=(q@k.transpose(-2,-1))*self.scale
 prob=mix(self,logits,'talk_pre').softmax(-1)
 return prob,mix(self,prob,'talk_post')

def matrices(self):
 return {n:(torch.diag(getattr(self,n)) if self.talking_arm=='TH1' else getattr(self,n)) for n in ('talk_pre','talk_post')}

def forward(self,x,x2,use_attn=True,inference_target_only=False):
 B,N,C=x2.shape
 if inference_target_only:
  q2,k2,v2=self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0);p,a=self_weights(self,q2,k2);self.attn=a;ad=self.attn_drop(a);o2=(ad@v2) if use_attn else v2;o2=self.proj_drop(self.proj(o2.transpose(1,2).reshape(B,N,C)));out=(None,o2,None,None)
 else:
  q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  q2,k2,v2=self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  qs,ks,vs=torch.cat((q,q2),0),torch.cat((k2,k),0),torch.cat((v2,v),0);p,a=self_weights(self,q,k);p2,a2=self_weights(self,q2,k2);ac=((qs@ks.transpose(-2,-1))*self.scale).softmax(-1);self.attn=a
  ad,a2d,acd=self.attn_drop(a),self.attn_drop(a2),self.attn_drop(ac)
  o,o2,oc=(ad@v,a2d@v2,acd@vs) if use_attn else (v,v2,vs)
  o=self.proj_drop(self.proj(o.transpose(1,2).reshape(B,N,C)));o2=self.proj_drop(self.proj(o2.transpose(1,2).reshape(B,N,C)));oc=self.proj_drop(self.proj(oc.transpose(1,2).reshape(2*B,N,C)));o3,o4=oc.split(B,0);out=(o,o2,o3,o4)
 if self.talking_log:
  with torch.no_grad():
   info={};eye=torch.eye(self.num_heads,device=a.device)
   for name,v in matrices(self).items():
    off=v*(1-eye);info[name]={'matrix':v.cpu().tolist(),'diagonal':v.diag().cpu().tolist(),'offdiagonal_Frobenius':float(off.norm()),'offdiagonal_to_total_norm':float(off.norm()/v.norm().clamp_min(1e-12)),'identity_difference_Frobenius':float((v-eye).norm())}
   info.update({'head_softmax_entropy_BEFORE_postmix':(-(p*p.clamp_min(1e-12).log()).sum(-1)).mean((0,2)).cpu().tolist(),'head_postmix_row_mass_mean':a.sum(-1).mean((0,2)).cpu().tolist(),'postmix_row_mass_min':float(a.sum(-1).min()),'postmix_row_mass_max':float(a.sum(-1).max()),'postmix_negative_weight_fraction':float((a<0).float().mean()),'postmix_weights_finite':bool(torch.isfinite(a).all())});self.last_diagnostics=info
 return out

def install(m,arm):
 assert arm in s.ARMS;att=m.blocks[0].attn;before=s.env().state_hash(m);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
 for name in ('talk_pre','talk_post'):att.register_parameter(name,nn.Parameter(torch.ones(att.num_heads) if arm=='TH1' else torch.eye(att.num_heads)))
 att.talking_arm=arm;att.talking_log=False;att.forward=types.MethodType(forward,att);assert base_hash(m)==before and torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
 added=16 if arm=='TH1' else 128;assert sum(p.numel() for p in m.parameters())==378011+added
 return {'base_initial_hash':before,'added_parameters':added,'total_parameters':378011+added,'new_parameter_hashes':{n:s.env().sha_array(getattr(att,n).detach().numpy()) for n in ('talk_pre','talk_post')},'initial_effective_matrices':{n:v.tolist() for n,v in matrices(att).items()},'no_randomdraws':True,'bias':False}
