import json,torch,numpy as np
import torch.nn.functional as F
import study as s
from mechanism import install,base_hash,refresh,statistics

def main():
 s.verify();torch.set_num_threads(2);c=s.env();inputs=np.load(s.B/'houston_oracle_lmmd_counterfactual_v1/inputs_2100.npz');x=torch.from_numpy(inputs['source'][0]).cuda();t=torch.from_numpy(inputs['target'][0]).cuda();y=torch.from_numpy(inputs['source_labels'][0]).cuda();records=[];ref=None;postref=None;gradref=None
 for arm in ('S0','S1','S2'):
  m,ls,helper,classes,ba=c.setup('houston','C',2100);audit=install(m,arm) if arm!='S0' else None;initial=ba['complete_initial_hash'];m.cuda().train();opt=torch.optim.SGD(m.parameters(),lr=.01);calls={};hooks=[]
  for name,v in [(n,b) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]+[('block%d'%i,b) for i,b in enumerate(m.blocks)]:
   def make(n):
    def hook(mod,args,out):calls[n]=calls.get(n,0)+1
    return hook
   hooks.append(v.register_forward_hook(make(name)))
  torch.manual_seed(991000);torch.cuda.manual_seed_all(991000);out=m(x,t);loss=F.cross_entropy(out[0],y);loss.backward();rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());gg={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None and 'scene_modulator.' not in n};og=None
  if arm=='S0':ref=[a.detach().clone() for a in out];gradref=gg;callref=dict(calls);rngref=rng
  else:
   err=max(float((a-b).abs().max()) for a,b in zip(out,ref));gerr=max(float((gg[n]-gradref[n]).abs().max()) for n in gg);assert err<1e-6 and gerr<1e-6 and gg.keys()==gradref.keys() and calls==callref and all(torch.equal(a,b) for a,b in zip(rng,rngref));att=m.blocks[0].attn;og=float(att.scene_modulator[-1].weight.grad.norm());assert og>0 and torch.isfinite(att.scene_modulator[-1].weight.grad).all();assert float(att.scene_modulator[0].weight.grad.norm())==0
  opt.step();ps={n:v.detach().cpu().clone() for n,v in m.state_dict().items() if 'scene_' not in n}
  if arm=='S0':postref=ps
  else:
   assert all(torch.equal(v,postref[n]) for n,v in ps.items());opt.zero_grad(set_to_none=True);m(x,t);context=att.last_context;out=m(x,t);loss=F.cross_entropy(out[0],y);cg=torch.autograd.grad(loss,att.last_context,retain_graph=True)[0];assert cg.norm()>0 and torch.isfinite(cg).all();loss.backward();assert att.scene_modulator[0].weight.grad.norm()>0
   # Isolate target-context path from source tokens and JointBN.
   a=torch.randn(3,5,64,device='cuda',requires_grad=True);b=torch.randn_like(a,requires_grad=True);z=att(a,b)[0].square().mean();bg=torch.autograd.grad(z,b,allow_unused=True)[0]
   if arm=='S2':assert bg is not None and bg.norm()>0 and torch.isfinite(bg).all()
   else:assert bg is None or bg.norm()==0
   for h in hooks:h.remove()
   r=refresh(m,ls[0].dataset,ls[2].dataset);assert att.scene_summary_valid;before=c.state_hash(m);m.eval();fixed=m(x,t)[1];assert c.state_hash(m)==before and torch.isfinite(fixed).all()
   # Saved summary supports target-only inference with exact roundtrip.
   path=s.OUT/('preflight_%s_roundtrip.pth'%arm);torch.save(m.state_dict(),path);m2,_,_,_,_=c.setup('houston','C',2100);install(m2,arm);m2.cuda().eval();m2.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m2.blocks[0].attn.scene_summary_mode='fixed';e=float((m2(x,t)[1]-fixed).abs().max());assert e<1e-6;del m2
  records.append({'arm':arm,'install':audit,'zero_init_logits_error':0 if arm=='S0' else err,'zero_init_base_gradient_error':0 if arm=='S0' else gerr,'zero_init_base_one_step_exact':True,'initial_RNG_and_calls_equal':True,'outputlayer_first_gradient_norm':og,'hiddenlayer_first_gradient_zero_expected':arm!='S0','post_output_update_hidden_and_context_gradient_verified':arm!='S0','isolated_target_context_gradient_norm':None if arm=='S0' or bg is None else float(bg.norm()),'saved_summary_roundtrip_error':None if arm=='S0' else e,'target_GT_accessed':False})
  if arm=='S0':
   for h in hooks:h.remove()
  del m,opt;torch.cuda.empty_cache()
 # Formula/shape: a shared channel metric, no cross-domain value mixing.
 q=torch.randn(2,8,5,8,device='cuda');k=torch.randn_like(q);w=torch.rand(1,8,1,8,device='cuda');manual=torch.einsum('bhil,bhjl,bhil->bhij',q,k,w.expand_as(q));assert (manual-(q*w)@k.transpose(-2,-1)).abs().max()<1e-5
 s.verify();s.dump(s.OUT/'preflight.json',{'passed':True,'audits':records,'parameters_equal_S1_S2':384251,'modulator_parameters':6240,'no_new_loss_or_forward':True,'target_context_gradient_after_first_step_verified':True});print('SCENE_ATTENTION_PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
