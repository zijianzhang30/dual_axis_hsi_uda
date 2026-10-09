import json,torch
import torch.nn.functional as F
import study as s
from mcc_loss import MinimumClassConfusionLoss
from diagnostics import diagnostic

def main():
 torch.set_num_threads(2);s.verify();rows=[]
 for seed in s.SEEDS:
  c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',seed);m=m.cuda().train();expected=json.loads((s.baseline(seed)/'config.json').read_text());assert audit['complete_initial_hash']==expected['initial_full_model_sha256']
  loss=MinimumClassConfusionLoss(2.);official=s.official_loss();assert sum(p.numel() for p in loss.parameters())==0
  z=torch.randn(128,7,device='cuda',requires_grad=True);l=loss(z.float());o=official(z.float());err=float((l-o).abs());assert err<1e-7;assert torch.equal(torch.autograd.grad(l,z,retain_graph=True)[0],torch.autograd.grad(o,z)[0]);diagnostic(z)
  x=torch.randn(14,1,48,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(14,device='cuda')%7;st={k:v.clone() for k,v in m.state_dict().items()};cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();heads=[];norms=[];counts={}
  def hc(mod,inp,out):heads.append(out)
  hook=m.nn1.register_forward_hook(hc)
  hooks=[m.norm.register_forward_hook(lambda mod,inp,out:norms.append(out))]
  for k,mod in m.named_modules():
   if isinstance(mod,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)):
    def make(k):
     def h(mod,inp,out):counts[k]=counts.get(k,0)+1
     return h
    hooks.append(mod.register_forward_hook(make(k)))
  out=m(x,t);assert len(heads)==4 and torch.equal(out[1],heads[1]) and len(norms)==4
  ce=F.cross_entropy(out[0],y);mcc=loss(out[1].float());tg=torch.autograd.grad(mcc,out[1],retain_graph=True)[0];pg=torch.autograd.grad(mcc,list(m.parameters()),retain_graph=True,allow_unused=True);assert torch.isfinite(tg).all() and tg.norm()>0;gn=sum(float(g.square().sum()) for g in pg if g is not None)**.5;assert gn>0
  opt=torch.optim.SGD(m.parameters(),lr=.01);ce.backward();grad0={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None};ceval=float(ce.detach());opt.step();update0={k:v.clone() for k,v in m.state_dict().items()};rngend=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());counts0=dict(counts)
  m.load_state_dict(st);m.zero_grad(set_to_none=True);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);heads.clear();norms.clear();counts.clear();again=m(x,t);other=F.cross_entropy(again[0],y)+0.*loss(again[1].float());other.backward()
  le=max(float((a-b).abs().max()) for a,b in zip(out,again));ge=max(float((p.grad-grad0[k]).abs().max()) for k,p in m.named_parameters() if k in grad0);assert le<1e-6 and ge<1e-6 and abs(float(other.detach())-ceval)<1e-6;opt.step();ue=max(float((m.state_dict()[k]-v).abs().max()) for k,v in update0.items());assert ue<1e-6 and counts0==counts and all(n==1 for n in counts.values());assert torch.equal(rngend[0],torch.get_rng_state()) and torch.equal(rngend[1],torch.cuda.get_rng_state())
  for h in hooks+[hook]:h.remove()
  assert sum(p.numel() for p in m.parameters())==expected['parameters']
  rows.append({'seed':seed,'official_loss_max_error':err,'lambda0_logits_error':le,'lambda0_gradient_error':ge,'lambda0_update_error':ue,'lambda0_loss_error':abs(float(other.detach())-ceval),'target_self_return_index':1,'target_logit_MCC_gradient_norm':float(tg.norm()),'main_MCC_gradient_norm':gn,'BN_calls_per_paired_forward':counts,'no_new_trainable_parameters':True,'optimizer_parameter_coverage':True,'target_labels_not_used':True})
 s.dump(s.OUT/'preflight.json',{'passed':True,'checks':rows});print('MCC_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
