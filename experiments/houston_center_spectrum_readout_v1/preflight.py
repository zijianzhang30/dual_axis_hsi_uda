import json,torch,numpy as np
import torch.nn.functional as F
import study as s
from readout import install,base_hash

def main():
 s.verify();torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);c=s.environment();audit=[]
 data=np.load(s.B/'houston_oracle_lmmd_counterfactual_v1/inputs_2100.npz');x=torch.from_numpy(data['source'][0]).cuda();t=torch.from_numpy(data['target'][0]).cuda();y=torch.from_numpy(data['source_labels'][0]).cuda()
 # Archived Oracle labels never accessed; only original augmented inputs/source labels used.
 for seed in s.SEEDS:
  reference=None;state_ref=None;grad_ref=None;initial=None
  for arm in ('R0','R1','R2'):
   m,ls,helper,classes,base=c.setup('houston','C',seed);initial0=c.state_hash(m);old=json.loads((s.baseline(seed)/'config.json').read_text());assert initial0==old['initial_full_model_sha256'];installed=install(m,arm,seed) if arm!='R0' else None;m.cuda().train();assert sum(p.numel() for p in m.parameters())==378011+(336 if arm!='R0' else 0)
   opt=torch.optim.SGD(m.parameters(),lr=.01);assert {id(p) for p in m.parameters()}=={id(p) for g in opt.param_groups for p in g['params']};calls={};hooks=[];captured=[]
   for n,mod in [(n,b) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]+[('block_%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
    def make(n):
     def h(mod,args,out):calls[n]=calls.get(n,0)+1
     return h
    hooks.append(mod.register_forward_hook(make(n)))
   if arm!='R0':hooks.append(m.center_readout.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone())))
   torch.manual_seed(995000);torch.cuda.manual_seed_all(995000);before_rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());out=m(x,t);loss=F.cross_entropy(out[0],y);loss.backward();after_rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());basegrad={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None and not n.startswith('center_readout.')};branch_grad=None
   if arm=='R0':reference=[z.detach().clone() for z in out];grad_ref=basegrad;calls_ref=dict(calls);rng_ref=after_rng
   else:
    err=max(float((a-b).abs().max()) for a,b in zip(reference,out));assert err<1e-6;assert calls==calls_ref;assert all(torch.equal(a,b) for a,b in zip(after_rng,rng_ref));assert basegrad.keys()==grad_ref.keys();grad_err=max(float((basegrad[n]-grad_ref[n]).abs().max()) for n in basegrad);assert grad_err<1e-6
    if arm=='R1':assert torch.equal(captured[0],x[:,0,:,6,6]) and torch.equal(captured[1],t[:,0,:,6,6])
    branch_grad=float(m.center_readout.weight.grad.norm());assert branch_grad>0 and torch.isfinite(m.center_readout.weight.grad).all()
   opt.step();post={n:v.detach().cpu().clone() for n,v in m.state_dict().items() if not n.startswith('center_readout.') and n!='center_projection'}
   if arm=='R0':state_ref=post
   else:assert all(torch.equal(v,state_ref[n]) for n,v in post.items());assert m.center_readout.weight.abs().max()>0
   before=c.state_hash(m);m.eval();z=m(x,t)[1];assert c.state_hash(m)==before
   if arm=='R1':
    for dims in ((-1,),(-2,),(-1,-2)):
     transformed=torch.flip(x,dims);assert torch.equal(transformed[:,0,:,6,6],x[:,0,:,6,6])
    for k in (1,2,3):assert torch.equal(torch.rot90(x,k,(-2,-1))[:,0,:,6,6],x[:,0,:,6,6])
   audit.append({'seed':seed,'arm':arm,'base_initial_sha256':initial0,'install':installed,'zero_initial_logits_max_error':0. if arm=='R0' else err,'base_gradient_max_error':0. if arm=='R0' else grad_err,'first_step_base_parameters_exactly_equal_R0':True,'branch_gradient_norm':branch_grad,'optimizer_includes_new_weights':True,'initial_random_stream_and_BN_Transformer_calls_equal':True,'eval_buffers_unchanged':True,'raw_center_is_actual_augmented_input':arm=='R1','target_GT_not_accessed':True})
   for h in hooks:h.remove()
   del m,opt;torch.cuda.empty_cache()
 s.verify();s.dump(s.OUT/'preflight.json',{'passed':True,'audits':audit,'independent_init_and_projection_RNG':True,'R2_algebraically_linear_control':True,'training_epochs_started':0});print('CENTER_READOUT_PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
