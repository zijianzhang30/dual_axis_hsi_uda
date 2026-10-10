import types
import torch
import torch.nn.functional as F
from torch import nn
import study as s

def base_hash(m):
 h=__import__('hashlib').sha256()
 for n,v in m.state_dict().items():
  if 'attention_log_scale' in n or 'cosine_base_scale' in n:continue
  h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def score(self,q,k):
 if self.attention_arm=='N1':base=(q@k.transpose(-2,-1))*self.scale;factor=self.attention_log_scale.exp()
 else:
  base=F.normalize(q,dim=-1,eps=1e-12)@F.normalize(k,dim=-1,eps=1e-12).transpose(-2,-1);factor=self.cosine_base_scale*self.attention_log_scale.exp()
 assert torch.isfinite(factor).all() and (factor>0).all(),'attention_scale_nonfinite_STOP'
 return base*factor.reshape(1,self.num_heads,1,1)

def forward(self,x,x2,use_attn=True,inference_target_only=False):
 B,N,C=x2.shape
 if inference_target_only:
  q2,k2,v2=self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  logits2=score(self,q2,k2);a2=logits2.softmax(-1);self.attn=a2;ad=self.attn_drop(a2);o2=(ad@v2) if use_attn else v2;o2=self.proj_drop(self.proj(o2.transpose(1,2).reshape(B,N,C)));out=(None,o2,None,None)
  q,k,logits,a=q2,k2,logits2,a2
 else:
  q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  q2,k2,v2=self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  qs,ks,vs=torch.cat((q,q2),0),torch.cat((k2,k),0),torch.cat((v2,v),0)
  logits=score(self,q,k);logits2=score(self,q2,k2);lc=(qs@ks.transpose(-2,-1))*self.scale
  a,a2,ac=logits.softmax(-1),logits2.softmax(-1),lc.softmax(-1);self.attn=a
  ad,a2d,acd=self.attn_drop(a),self.attn_drop(a2),self.attn_drop(ac)
  o,o2,oc=(ad@v,a2d@v2,acd@vs) if use_attn else (v,v2,vs)
  o=self.proj_drop(self.proj(o.transpose(1,2).reshape(B,N,C)));o2=self.proj_drop(self.proj(o2.transpose(1,2).reshape(B,N,C)));oc=self.proj_drop(self.proj(oc.transpose(1,2).reshape(2*B,N,C)));o3,o4=oc.split(B,0);out=(o,o2,o3,o4)
 if self.attention_log:
  with torch.no_grad():
   centered=logits-logits.mean(-1,keepdim=True);factor=self.attention_log_scale.exp()*(self.cosine_base_scale if self.attention_arm=='N2' else 1)
   self.last_diagnostics={'head_effective_scales':factor.cpu().tolist(),'head_log_scale_parameters':self.attention_log_scale.detach().cpu().tolist(),'head_Q_norm_mean':q.norm(dim=-1).mean((0,2)).cpu().tolist(),'head_K_norm_mean':k.norm(dim=-1).mean((0,2)).cpu().tolist(),'head_row_centered_score_RMS':centered.square().mean((0,2,3)).sqrt().cpu().tolist(),'head_attention_entropy':(-(a*a.clamp_min(1e-12).log()).sum(-1)).mean((0,2)).cpu().tolist()}
 return out

def install(m,arm):
 assert arm in s.ARMS;att=m.blocks[0].attn;original=s.env().state_hash(m);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all();att.register_parameter('attention_log_scale',nn.Parameter(torch.zeros(att.num_heads)))
 if arm=='N2':
  calibration=__import__('json').loads((s.OUT/'source_calibration.json').read_text());att.register_buffer('cosine_base_scale',torch.tensor(calibration['s_h0'],dtype=torch.float32))
 att.attention_arm=arm;att.attention_log=False;att.forward=types.MethodType(forward,att)
 assert base_hash(m)==original and torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
 assert sum(p.numel() for p in m.parameters())==378019
 return {'base_initial_hash':original,'added_parameter_initial_sha256':s.env().sha_array(att.attention_log_scale.detach().numpy()),'added_trainable_parameters':8,'total_parameters':378019,'s_h0':att.cosine_base_scale.tolist() if arm=='N2' else None,'base_and_main_RNG_preserved':True}
