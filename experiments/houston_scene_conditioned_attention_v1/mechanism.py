import copy, types
import numpy as np
import torch
from torch import nn
import study as s

def base_hash(m):
 h=__import__('hashlib').sha256()
 for n,v in m.state_dict().items():
  if any(x in n for x in ('scene_modulator.','scene_summary','scene_summary_valid')):continue
  h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def statistics(x):
 z=x[:,1:,:].reshape(-1,x.shape[-1]);return torch.cat((z.mean(0),z.var(0,unbiased=False).clamp_min(0).sqrt()))

def forward(self,x,x2,use_attn=True,inference_target_only=False):
 B,N,C=x2.shape
 if self.scene_summary_mode=='fixed':
  assert bool(self.scene_summary_valid),'fixed summary must be recomputed from current checkpoint'
  context=self.scene_summary
 else:
  context=statistics(x if self.scene_arm=='S1' and x is not None else x2)
 self.last_context=context
 m=1+.1*torch.tanh(self.scene_modulator(context))
 mh=m.reshape(1,self.num_heads,1,C//self.num_heads)
 qkv2=self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4) if inference_target_only else None
 if inference_target_only:
  q2,k2,v2=qkv2.unbind(0);a2=((q2*mh)@k2.transpose(-2,-1))*self.scale;a2=a2.softmax(-1);self.attn=a2
  o2=(self.attn_drop(a2)@v2) if use_attn else v2
  o2=self.proj_drop(self.proj(o2.transpose(1,2).reshape(B,N,C)));result=(None,o2,None,None)
  ent=-(a2.detach()*a2.detach().clamp_min(1e-12).log()).sum(-1).mean()
 else:
  q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  q2,k2,v2=self.qkv(x2).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
  qs,ks,vs=torch.cat((q,q2),0),torch.cat((k2,k),0),torch.cat((v2,v),0)
  a=((q*mh)@k.transpose(-2,-1))*self.scale;a2=((q2*mh)@k2.transpose(-2,-1))*self.scale
  ac=(qs@ks.transpose(-2,-1))*self.scale
  a,a2,ac=a.softmax(-1),a2.softmax(-1),ac.softmax(-1);self.attn=a
  ent=-(a.detach()*a.detach().clamp_min(1e-12).log()).sum(-1).mean()
  # Preserve exact original dropout order and unused coupled computation.
  ad,a2d,acd=self.attn_drop(a),self.attn_drop(a2),self.attn_drop(ac)
  o,o2,oc=(ad@v,a2d@v2,acd@vs) if use_attn else (v,v2,vs)
  o=self.proj_drop(self.proj(o.transpose(1,2).reshape(B,N,C)))
  o2=self.proj_drop(self.proj(o2.transpose(1,2).reshape(B,N,C)))
  oc=self.proj_drop(self.proj(oc.transpose(1,2).reshape(2*B,N,C)));o3,o4=oc.split(B,0);result=(o,o2,o3,o4)
 if self.scene_log:
  self.last_diagnostics={'modulation_min':float(m.detach().min()),'modulation_max':float(m.detach().max()),'modulation_abs_deviation_mean':float((m.detach()-1).abs().mean()),'source_attention_entropy' if not inference_target_only else 'target_attention_entropy':float(ent),'summary_mean_norm':float(context[:64].detach().norm()),'summary_std_norm':float(context[64:].detach().norm())}
 return result

def install(m,arm,seed=2100):
 assert arm in s.ARMS
 att=m.blocks[0].attn;original=s.env().state_hash(m);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(seed+181000);att.scene_modulator=nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Linear(32,64));nn.init.zeros_(att.scene_modulator[-1].weight);nn.init.zeros_(att.scene_modulator[-1].bias)
 att.register_buffer('scene_summary',torch.zeros(128));att.register_buffer('scene_summary_valid',torch.tensor(False))
 att.scene_arm=arm;att.scene_summary_mode='batch';att.scene_log=False;att.forward=types.MethodType(forward,att)
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())) and base_hash(m)==original
 assert sum(p.numel() for p in att.scene_modulator.parameters())==6240
 return {'base_initial_hash':original,'modulator_initial_hash':s.env().state_hash(att.scene_modulator),'additional_trainable_parameters':6240,'total_trainable_parameters':sum(p.numel() for p in m.parameters()),'private_RNG_preserved':True}

class Captured(Exception):pass

def summary_loader(dataset,positions):
 d=copy.copy(dataset);d.indices=dataset.indices[np.asarray(positions)];d.labels=np.zeros(len(d.indices),dtype=np.int64);d.label=np.zeros_like(dataset.label);d.data_aug=False
 return torch.utils.data.DataLoader(d,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(184000))

@torch.no_grad()
def refresh(m,source_dataset,target_dataset):
 att=m.blocks[0].attn;selected=json_load_cohorts();name='source' if att.scene_arm=='S1' else 'target';ds=source_dataset if name=='source' else target_dataset
 loader=summary_loader(ds,selected[name]['positions']);was=m.training;m.eval();before=s.env().state_hash(m);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();total=0;su=torch.zeros(64,device='cuda',dtype=torch.float64);sq=su.clone();last=[]
 def capture(mod,args):
  last.append(args[1][:,1:,:].detach());raise Captured()
 hook=att.register_forward_pre_hook(capture)
 try:
  for x,_ in loader:
   x=x.cuda()
   try:m(x,x)
   except Captured:pass
   z=last.pop().reshape(-1,64).double();total+=len(z);su+=z.sum(0);sq+=(z*z).sum(0)
 finally:hook.remove()
 assert s.env().state_hash(m)==before and torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
 mean=su/total;std=(sq/total-mean.square()).clamp_min(0).sqrt();context=torch.cat((mean,std)).float();assert torch.isfinite(context).all();att.scene_summary.copy_(context);att.scene_summary_valid.fill_(True);att.scene_summary_mode='fixed';m.train(was)
 return {'domain':name,'samples':len(selected[name]['positions']),'token_observations':total,'summary':context.cpu().tolist(),'summary_cohort_indices_hash':selected[name]['indices_sha256'],'eval_frozen_BN_and_main_RNG_preserved':True}

def json_load_cohorts():return __import__('json').loads((s.OUT/'summary_cohorts.json').read_text())
