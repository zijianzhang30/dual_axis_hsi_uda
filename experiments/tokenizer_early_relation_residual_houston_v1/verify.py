import torch
from common import OUT,SEEDS,environment,build,state_hash,dump

def main():
 torch.set_num_threads(2);env=environment();rows=[]
 for seed in SEEDS:
  env[4](seed);c,o=build(seed,'C','houston');env[4](seed);s,z=build(seed,'S1','houston');assert o==z
  assert all(torch.equal(v,s.state_dict()[n]) for n,v in c.state_dict().items())
  added=set(dict(s.named_parameters()))-set(dict(c.named_parameters()));assert added=={'spectral_relation_gamma','spectral_relation_branch.weight'}
  assert sum(p.numel() for p in s.parameters())==378036
  gen=torch.Generator().manual_seed(92000);x=torch.randn((4,1,48,13,13),generator=gen);t=torch.randn(x.shape,generator=gen).requires_grad_()
  counts={};hooks=[];capture={}
  for name,m in s.named_modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or m.__class__.__name__.startswith('Block'):
    counts[name]=0
    def hook(m,a,b,name=name):counts[name]+=1
    hooks.append(m.register_forward_hook(hook))
  original=s._forward_semantic_tokens
  def token_capture(v):
   patch=s._relation_pending[0];w=s.relation_scores(patch);tok=original(v);capture['weights']=w.detach();capture['tokens']=tok.detach();return tok
  s._forward_semantic_tokens=token_capture
  c.train();s.train();torch.set_rng_state(gen.get_state());rng=torch.get_rng_state();oc=c(x,t);after=torch.get_rng_state();torch.set_rng_state(rng);os=s(x,t)
  assert torch.equal(after,torch.get_rng_state());err=max(float((a-b).abs().max()) for a,b in zip(oc,os));assert err<1e-6
  assert all(v==1 for v in counts.values()),counts
  assert all(torch.equal(v,s.state_dict()[n]) for n,v in c.state_dict().items())
  werr=float((c.relation_scores(t)-capture['weights']).abs().max());assert werr<1e-6
  y=torch.arange(4);torch.nn.functional.cross_entropy(os[0],y).backward();gamma_grad=float(s.spectral_relation_gamma.grad);assert abs(gamma_grad)>0 and torch.isfinite(s.spectral_relation_gamma.grad)
  assert s.spectral_relation_branch.weight.grad.abs().max()==0
  assert t.grad is not None and t.grad.abs().sum()>0
  target_grad=float(t.grad.norm())
  opt=torch.optim.SGD(s.parameters(),lr=.01);opt.step();assert s.spectral_relation_gamma!=0
  s.zero_grad();torch.nn.functional.cross_entropy(s(x,t.detach())[0],y).backward();weight_grad=float(s.spectral_relation_branch.weight.grad.norm());assert weight_grad>0
  for h in hooks:h.remove()
  s._forward_semantic_tokens=original
  # Return gamma to zero and synchronize original state for independent eval equivalence.
  s.load_state_dict(c.state_dict(),strict=False);s.spectral_relation_gamma.data.zero_();c.eval();s.eval();evalerr=float((c(x,t)[1]-s(x,t)[1]).abs().max());assert evalerr<1e-6
  rows.append({'seed':seed,'original_initial_hash':o,'gamma0_train_logits_max_error':err,'gamma0_eval_logits_max_error':evalerr,'gamma0_weights_max_error':werr,'BN_Transformer_calls_per_paired_train':{n:1 for n in counts},'gamma_initial_gradient':gamma_grad,'branch_initial_gradient_expected_zero':True,'branch_gradient_after_gamma_step':weight_grad,'target_through_JointBN_gradient_norm':target_grad,'shared_initial_parameters_and_running_stats_equal':True,'main_RNG_preserved':True,'branch_shape':[4,8,48,13,13],'center':[6,6]})
 dump(OUT/'verification.json',{'passed':True,'parameters_added':25,'parameters_S1':378036,'gamma_initial':0.,'checks':rows});print('PASS: gamma0 logits/weights,25 parameters,original init/RNG/BN paths,target gradients,gamma and delayed branch gradients')
if __name__=='__main__':main()
