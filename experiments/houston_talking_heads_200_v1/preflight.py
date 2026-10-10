import json
import numpy as np,torch
import torch.nn.functional as F
import study as s
from attention import install,mix,matrices

def main():
 s.verify();torch.set_num_threads(2);c=s.env();data=np.load(s.B/'houston_oracle_lmmd_counterfactual_v1/inputs_2100.npz');x=torch.from_numpy(data['source'][0]).cuda();t=torch.from_numpy(data['target'][0]).cuda();y=torch.from_numpy(data['source_labels'][0]).cuda();records=[]
 for arm in ('TH0','TH1','TH2'):
  m,ls,helper,classes,ba=c.setup('houston','C',2100);audit=install(m,arm) if arm!='TH0' else None;m.cuda().train();opt=torch.optim.SGD(m.parameters(),lr=.01);assert {id(p) for p in m.parameters()}=={id(p) for g in opt.param_groups for p in g['params']};calls={};hooks=[]
  for name,v in [(n,v) for n,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]+[('block%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
   def make(n):
    def h(mod,args,out):calls[n]=calls.get(n,0)+1
    return h
   hooks.append(v.register_forward_hook(make(name)))
  torch.manual_seed(997000);torch.cuda.manual_seed_all(997000);out=m(x,t);assert all(torch.isfinite(a).all() for a in out);loss=F.cross_entropy(out[0],y);loss.backward();rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());gg={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None and 'talk_pre' not in n and 'talk_post' not in n};assert all(torch.isfinite(g).all() for g in gg.values());ng={}
  if arm=='TH0':ref=[a.detach().clone() for a in out];gradref=gg;callsref=dict(calls);rngref=rng
  else:
   err=max(float((a-b).abs().max()) for a,b in zip(ref,out));gerr=max(float((gg[n]-gradref[n]).abs().max()) for n in gg);assert err<1e-6 and gerr<1e-6 and gg.keys()==gradref.keys() and calls==callsref and all(torch.equal(a,b) for a,b in zip(rng,rngref));att=m.blocks[0].attn
   for name in ('talk_pre','talk_post'):
    g=getattr(att,name).grad;assert g is not None and torch.isfinite(g).all() and g.norm()>0;ng[name]={'gradient_norm':float(g.norm())}
    if arm=='TH2':off=g*(1-torch.eye(8,device=g.device));assert off.norm()>0;ng[name]['offdiagonal_gradient_norm']=float(off.norm())
  opt.step();post={n:v.detach().cpu().clone() for n,v in m.state_dict().items() if 'talk_pre' not in n and 'talk_post' not in n}
  if arm=='TH0':postref=post
  else:
   assert all(torch.equal(v,postref[n]) for n,v in post.items());assert any((v-torch.eye(8,device=v.device)).abs().max()>0 for v in matrices(att).values())
   if arm=='TH2':assert all((v*(1-torch.eye(8,device=v.device))).norm()>0 for v in matrices(att).values())
  for h in hooks:h.remove()
  m.eval();before=c.state_hash(m);z=m(x,t)[1];assert c.state_hash(m)==before and torch.isfinite(z).all();roundtrip=None
  if arm!='TH0':
   path=s.OUT/('preflight_%s_roundtrip.pth'%arm);torch.save(m.state_dict(),path);m2,_,_,_,_=c.setup('houston','C',2100);install(m2,arm);m2.cuda().eval();m2.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));roundtrip=float((m2(x,t)[1]-z).abs().max());assert roundtrip<1e-6;del m2
   # Independent manual contraction confirms head-axis,not query/key mixing.
   v=torch.arange(2*8*5*5,device='cuda',dtype=torch.float32).reshape(2,8,5,5)/1000
   for name,p in matrices(att).items():
    actual=mix(att,v,name);manual=torch.stack([sum(v[:,i]*p[o,i] for i in range(8)) for o in range(8)],1);assert (actual-manual).abs().max()<1e-6
  records.append({'arm':arm,'install':audit,'identity_logits_max_error':0 if arm=='TH0' else err,'identity_base_gradient_max_error':0 if arm=='TH0' else gerr,'identity_base_one_step_exact':True,'new_parameter_gradients':ng,'head_axis_manual_contraction_passed':arm!='TH0','main_RNG_and_BN_Transformer_calls_equal':True,'eval_buffers_unchanged':True,'save_load_logits_error':roundtrip,'target_GT_not_accessed':True})
  del m,opt;torch.cuda.empty_cache()
 s.verify();s.dump(s.OUT/'preflight.json',{'passed':True,'audits':records,'projection_order':'headmix->softmax_key->headmix->originaldropout->sameheadV','no_extra_forward_or_loss':True,'TH1_TH2_parametercounts':{'TH1':378027,'TH2':378139},'not_capacitymatched':True});print('TALKING_HEADS_PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
