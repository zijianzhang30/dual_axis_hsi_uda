from contextlib import contextmanager
import copy,hashlib
import numpy as np,torch
from torch import nn
import torch.nn.functional as F

class RawCompanion(torch.utils.data.Dataset):
 def __init__(self,base,path):
  self.base=base;self.raw=np.load(path,mmap_mode='r');assert not(base.flip_augmentation or base.radiation_augmentation or base.mixture_augmentation)
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  before=np.random.get_state();x,y=self.base[i];after=np.random.get_state();a,b=self.base.indices[i];r=self.raw[a-6:a+7,b-6:b+7]
  try:
   np.random.set_state(before)
   if self.base.data_aug:r=self.base.hsi_augment(r)
  finally:np.random.set_state(after)
  return x,y,torch.from_numpy(np.asarray(np.copy(r).transpose(2,0,1),dtype=np.float32)).unsqueeze(0)
def wrap(loader,path):
 return torch.utils.data.DataLoader(RawCompanion(loader.dataset,path),batch_size=loader.batch_size,sampler=loader.sampler,num_workers=loader.num_workers,drop_last=loader.drop_last,worker_init_fn=loader.worker_init_fn,generator=loader.generator,pin_memory=loader.pin_memory,collate_fn=loader.collate_fn)

def mask_batch(n,generator,device):
 rank=torch.rand(n,48,generator=generator).argsort(1);mask=torch.zeros(n,48,dtype=torch.bool);mask.scatter_(1,rank[:,:24],True);assert (mask.sum(1)==24).all();return mask.to(device)
def visible_input(raw,mask):
 assert raw.shape[1:]==(1,48,13,13);v=raw.double()*~mask[:,None,:,None,None];norm=v.square().sum(2,keepdim=True).sqrt();x=v/torch.where(norm>0,norm,torch.ones_like(norm));assert torch.isfinite(x).all();return x.float()
def full_normalize(raw):
 x=raw.double();n=x.square().sum(2,keepdim=True).sqrt();return (x/torch.where(n>0,n,torch.ones_like(n))).float()

def snapshots(m):
 return {n:{'mean':b.running_mean.detach().clone(),'var':b.running_var.detach().clone(),'tracked':b.num_batches_tracked.detach().clone()} for n,b in m.named_modules() if isinstance(b,(nn.BatchNorm2d,nn.BatchNorm3d))}
@contextmanager
def frozen_BN(m,snapshot):
 old={};mods=dict(m.named_modules())
 for n,v in snapshot.items():
  b=mods[n];old[n]=(b.running_mean,b.running_var,b.num_batches_tracked,b.training);b.running_mean=v['mean'];b.running_var=v['var'];b.num_batches_tracked=v['tracked'];b.training=False
 try:yield
 finally:
  for n,v in old.items():b=mods[n];b.running_mean,b.running_var,b.num_batches_tracked,b.training=v

def norm(gs):return sum(float(x.square().sum()) for x in gs if x is not None)**.5
class Decoder(nn.Module):
 def __init__(self):
  super().__init__();self.mlp=nn.Sequential(nn.Linear(258,32),nn.ReLU(),nn.Linear(32,48));v=torch.linspace(-1,1,13);a,b=torch.meshgrid(v,v,indexing='ij');self.register_buffer('coords',torch.stack((a,b),-1).reshape(169,2))
 def forward(self,tokens):
  assert tokens.shape[1:]==(4,64);b=len(tokens);x=torch.cat([tokens.flatten(1)[:,None,:].expand(b,169,256),self.coords[None].expand(b,169,2)],-1);return self.mlp(x).transpose(1,2).reshape(b,48,13,13)
def decoder(seed=2100):
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
 with torch.random.fork_rng(devices=[]):torch.set_rng_state(torch.Generator().manual_seed(seed+159000).get_state());d=Decoder()
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()));assert sum(p.numel() for p in d.parameters())==9872;return d

def target_values(m,full,arm):
 with torch.no_grad():
  v=full[:,0] if arm=='E1' else m.relation_scores.relations(full)
  assert v.shape[1:]==(48,13,13)
  if arm=='E2':assert v[:,:,6,6].abs().max()==0
  return v.detach()
def score(pred,target,mask,mean,std):
 # Same maskedbands/noncenter positions in E1 andE2; alltargets detached.
 scaled=(target-mean[None,:,None,None])/std[None,:,None,None];positions=mask[:,:,None,None].expand_as(target).clone();positions[:,:,6,6]=False;assert int(positions.sum())==len(mask)*24*168
 loss=F.huber_loss(pred,scaled,reduction='none',delta=1.);assert torch.isfinite(loss).all();return loss[positions].mean()

def auxiliary(m,d,raws,rawt,fulls,fullt,ms,mt,oldbn,arm,mean,std,seed):
 cap=[]
 def capture(mod,args,out):cap.append(out[:,1:])
 h=m.norm.register_forward_hook(capture);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();xs,xt=visible_input(raws,ms),visible_input(rawt,mt)
 try:
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(seed);torch.cuda.manual_seed(seed)
   with frozen_BN(m,oldbn):m(xs,xt)
   assert len(cap)==4;ps,pt=d(cap[0]),d(cap[1])
 finally:h.remove()
 assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
 ys,yt=target_values(m,fulls,arm),target_values(m,fullt,arm);ls,lt=score(ps,ys,ms,mean,std),score(pt,yt,mt,mean,std);return .5*(ls+lt),{'source_loss':float(ls.detach()),'target_loss':float(lt.detach())},(ps,pt)
