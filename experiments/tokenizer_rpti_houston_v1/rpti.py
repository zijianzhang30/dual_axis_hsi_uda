import inspect,textwrap
from types import MethodType
import torch
from torch import nn
EPS=1e-6

def provenance(weights):
 w=weights.detach();hat=w/(w.norm(dim=-1,keepdim=True)+EPS);r=hat@hat.transpose(-2,-1)
 assert r.shape==(len(w),4,4) and torch.isfinite(r).all()
 return r

def bias(attention,domain):
 r=attention._rpti_cache[domain];b=r.new_zeros((len(r),5,5));b[:,1:,1:]=r
 return attention.beta[None,:,None,None]*b[:,None,:,:]

def before_softmax(self,logits,domain):
 out=logits+bias(self,domain)
 if self._rpti_record:
  p=out.softmax(-1);entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1).mean((0,2))
  self._rpti_entropy[domain]=entropy.detach().cpu().tolist()
 return out

def semantic(self,features):
 patch=self._relation_pending.pop(0);w=self.relation_scores(patch)
 self._rpti_cache.append(provenance(w))
 if self._rpti_record:self._rpti_last_R=self._rpti_cache[-1].detach()
 return torch.einsum('bln,bcn->blc',w,features.flatten(2))

def forward(self,source,target,*args,**kwargs):
 self._rpti_cache.clear()
 for block in self.blocks:block.attn._rpti_record=self._rpti_record
 return self._rpti_original_forward(self,source,target,*args,**kwargs)

def install(model):
 model._rpti_cache=[];model._rpti_record=False
 model._forward_semantic_tokens=MethodType(semantic,model)
 model._rpti_original_forward=model.forward.__func__;model.forward=MethodType(forward,model)
 for block in model.blocks:
  att=block.attn;att.beta=nn.Parameter(torch.zeros(att.num_heads));att._rpti_cache=model._rpti_cache;att._rpti_record=False;att._rpti_entropy={}
  att._rpti_before=MethodType(before_softmax,att)
  code=textwrap.dedent(inspect.getsource(att.forward.__func__))
  code=code.replace('attn2 = attn2.softmax(dim=-1)','attn2 = self._rpti_before(attn2,1).softmax(dim=-1)').replace('attn = attn.softmax(dim=-1)','attn = self._rpti_before(attn,0).softmax(dim=-1)')
  namespace={'torch':torch};exec(code,namespace);att.forward=MethodType(namespace['forward'],att)

def diagnostic(model):
 r=getattr(model,'_rpti_last_R',None)
 if r is None:return {}
 mask=~torch.eye(4,dtype=torch.bool,device=r.device);v=r[:,mask]
 return {'R_min':float(v.min()),'R_mean':float(v.mean()),'R_max':float(v.max()),'R_std':float(v.std()),'self_attention_entropy_nats':[{str(k):z for k,z in b.attn._rpti_entropy.items()} for b in model.blocks]}
