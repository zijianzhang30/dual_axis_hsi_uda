import torch,math
from torch import nn
from torch.autograd import Function
from types import MethodType

class Reverse(Function):
 @staticmethod
 def forward(ctx,x,coeff):ctx.coeff=coeff;return x*1.
 @staticmethod
 def backward(ctx,g):return -ctx.coeff*g,None

def build_discriminator(seed):
 with torch.random.fork_rng(devices=[]):
  torch.set_rng_state(torch.Generator().manual_seed(seed+101000).get_state())
  d=nn.Sequential(nn.Linear(7*64,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())
 return d

def coefficient(step):return 2/(1+math.exp(-step/1000.))-1

def loss(discriminator,features,logits,coeff):
 fs,ft=features;gs,gt=logits
 if fs.ndim==3:
  b,k,d=fs.shape;assert k==4 and ft.shape==fs.shape
  fs=fs.reshape(b*k,d);ft=ft.reshape(b*k,d);gs=gs.repeat_interleave(k,0);gt=gt.repeat_interleave(k,0)
 f=torch.cat((fs,ft));p=torch.cat((gs,gt)).softmax(1).detach();u=torch.bmm(p.unsqueeze(2),f.unsqueeze(1)).flatten(1)
 scores=discriminator(Reverse.apply(u,coeff));label=torch.cat((torch.ones(len(fs),1,device=f.device),torch.zeros(len(ft),1,device=f.device)))
 return torch.nn.functional.binary_cross_entropy(scores,label),{'domain_accuracy':float(((scores.detach()>=.5)==label.bool()).float().mean()*100),'features_norm':float(f.detach().norm(dim=-1).mean()),'features_channel_variance':float(f.detach().var(0,unbiased=False).mean()),'conditional_shape':list(u.shape)}

def capture(m):
 bucket={'T':[],'Z':[]};original=m._forward_semantic_tokens
 def tokens(self,features):
  t=original(features);bucket['T'].append(t);return t
 m._forward_semantic_tokens=MethodType(tokens,m)
 h=m.norm.register_forward_hook(lambda mod,inp,out:bucket['Z'].append(out[:,0]))
 def clear():bucket['T'].clear();bucket['Z'].clear()
 def remove():h.remove();m._forward_semantic_tokens=original
 return bucket,clear,remove
