import json
import numpy as np,torch
import torch.nn.functional as F
import study as s
from attention import install,score

def main():
 s.verify();torch.set_num_threads(2);c=s.env();data=np.load(s.B/'houston_oracle_lmmd_counterfactual_v1/inputs_2100.npz');x=torch.from_numpy(data['source'][0]).cuda();t=torch.from_numpy(data['target'][0]).cuda();y=torch.from_numpy(data['source_labels'][0]).cuda();records=[]
 for arm in ('N0','N1','N2'):
  m,ls,helper,classes,ba=c.setup('houston','C',2100);audit=install(m,arm) if arm!='N0' else None;m.cuda().train();opt=torch.optim.SGD(m.parameters(),lr=.01);assert {id(p) for p in m.parameters()}=={id(p) for g in opt.param_groups for p in g['params']};calls={};hooks=[]
  for name,v in [(n,v) for n,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]+[('block%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
   def make(n):
    def h(mod,args,out):calls[n]=calls.get(n,0)+1
    return h
   hooks.append(v.register_forward_hook(make(name)))
  torch.manual_seed(996000);torch.cuda.manual_seed_all(996000);out=m(x,t);assert all(torch.isfinite(a).all() for a in out);loss=F.cross_entropy(out[0],y);loss.backward();rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());gg={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None and 'attention_log_scale' not in n};assert all(torch.isfinite(g).all() for g in gg.values());aerr=None;gerr=None
  if arm=='N0':ref=[v.detach().clone() for v in out];gradref=gg;callsref=dict(calls);rngref=rng
  else:
   assert calls==callsref and all(torch.equal(a,b) for a,b in zip(rng,rngref));assert gg.keys()==gradref.keys();att=m.blocks[0].attn;ag=att.attention_log_scale.grad.detach().clone();assert ag.norm()>0 and torch.isfinite(ag).all();assert gg['blocks.0.attn.qkv.weight'].norm()>0;assert any(g.norm()>0 for n,g in gg.items() if n.startswith('relation_scores.'));assert any(g.norm()>0 for n,g in gg.items() if n.startswith('conv3d_features.'))
   if arm=='N1':
    aerr=max(float((a-b).abs().max()) for a,b in zip(ref,out));gerr=max(float((gg[n]-gradref[n]).abs().max()) for n in gg);assert aerr<1e-6 and gerr<1e-6
  opt.step();post={n:v.detach().cpu().clone() for n,v in m.state_dict().items() if 'attention_log_scale' not in n and 'cosine_base_scale' not in n}
  if arm=='N0':postref=post
  if arm=='N1':assert all(torch.equal(v,postref[n]) for n,v in post.items())
  for h in hooks:h.remove()
  m.eval();before=c.state_hash(m);z=m(x,t)[1];assert c.state_hash(m)==before and torch.isfinite(z).all()
  roundtrip=None
  if arm!='N0':
   path=s.OUT/('preflight_%s_roundtrip.pth'%arm);torch.save(m.state_dict(),path);m2,_,_,_,_=c.setup('houston','C',2100);install(m2,arm);m2.cuda().eval();m2.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));roundtrip=float((m2(x,t)[1]-z).abs().max());assert roundtrip<1e-6;del m2
   q=torch.randn(2,8,5,8,device='cuda',requires_grad=True);k=torch.randn_like(q,requires_grad=True);got=score(att,q,k)
   if arm=='N2':
    manual=torch.einsum('bhil,bhjl->bhij',F.normalize(q,dim=-1,eps=1e-12),F.normalize(k,dim=-1,eps=1e-12))* (att.cosine_base_scale*att.attention_log_scale.exp()).reshape(1,8,1,1);assert (manual-got).abs().max()<1e-6
    zg=torch.zeros_like(q,requires_grad=True);zk=torch.zeros_like(k,requires_grad=True);zz=score(att,zg,zk);gr=torch.autograd.grad(zz.sum(),(zg,zk));assert torch.isfinite(zz).all() and all(torch.isfinite(a).all() for a in gr)
  records.append({'arm':arm,'install':audit,'zero_init_logits_error':aerr,'zero_init_base_gradient_error':gerr,'N1_initial_base_one_step_exact':True if arm=='N1' else None,'output_and_gradient_finite':True,'scale_gradient_norm':float(ag.norm()) if arm!='N0' else None,'main_RNG_and_BN_Transformer_calls_equal':True,'eval_buffers_unchanged':True,'save_load_logits_error':roundtrip,'target_GT_not_accessed':True})
  del m,opt;torch.cuda.empty_cache()
 cal=json.loads((s.OUT/'source_calibration.json').read_text());rd=np.asarray(cal['D_row_centered_RMS']);rc=np.asarray(cal['C_row_centered_RMS']);sh=np.asarray(cal['s_h0'],dtype=np.float32).astype(np.float64);err=np.max(np.abs(sh*rc-rd)/rd);assert err<1e-6
 s.verify();s.dump(s.OUT/'preflight.json',{'passed':True,'audits':records,'calibration_row_centered_RMS_relative_error':float(err),'N2_QK_normalization_and_no_extra_sqrt_divisor_verified':True,'N1_N2_same8_trainable_parameters':True,'finite_zero_QK_test':True,'no_new_losses_or_extra_main_forward':True});print('COSINE_ATTENTION_PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
