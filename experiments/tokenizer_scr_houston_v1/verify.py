import torch,json
from common import OUT,SEEDS,environment,build,state_hash,dump
from scr import compatibility,refined_weights

def main():
 torch.set_num_threads(2);env=environment();rows=[]
 for seed in SEEDS:
  env[4](seed);c,o=build(seed,'C','houston');env[4](seed);k,z=build(seed,'K1','houston');assert o==z
  assert all(torch.equal(v,k.state_dict()[n]) for n,v in c.state_dict().items());added=set(dict(k.named_parameters()))-set(dict(c.named_parameters()));assert added=={'scr_beta'} and k.scr_beta.numel()==4
  gen=torch.Generator().manual_seed(92000);x=torch.randn((4,1,48,13,13),generator=gen);t=x.flip(-1)
  recorded={};hooks=[];counts={}
  def cb(m,a,b):recorded['V']=b.detach()
  hooks.append(k.conv2d_features[2].register_forward_hook(cb))
  for name,m in k.named_modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or m.__class__.__name__.startswith('Block'):
    counts[name]=0
    def hook(m,a,b,name=name):counts[name]+=1
    hooks.append(m.register_forward_hook(hook))
  c.train();k.train();torch.manual_seed(42);rng=torch.get_rng_state();oc=c(x,t);after=torch.get_rng_state();torch.set_rng_state(rng);ok=k(x,t)
  assert torch.equal(after,torch.get_rng_state());err=max(float((a-b).abs().max()) for a,b in zip(oc,ok));assert err<1e-6
  assert all(v==1 for v in counts.values());values=recorded['V'][4:] # paired CNN outputsource+target
  wc=c.relation_scores(t);wk,sem=refined_weights(k,t,values);werr=float((wc-wk).abs().max());assert werr<1e-6
  tc=torch.einsum('bln,bcn->blc',wc,values.flatten(2));tk=torch.einsum('bln,bcn->blc',wk,values.flatten(2));terr=float((tc-tk).abs().max());assert terr<1e-6
  unit=torch.nn.functional.normalize(values,dim=1,eps=1e-6);center=unit[:,:,6,6];explicit=(unit.flatten(2)*center[:,:,None]).sum(1);assert torch.equal(sem,explicit);assert torch.isfinite(sem).all()
  torch.nn.functional.cross_entropy(ok[0],torch.arange(4)).backward();beta_grad=k.scr_beta.grad.detach().tolist();assert k.scr_beta.grad.abs().sum()>0 and torch.isfinite(k.scr_beta.grad).all()
  # Bias-only autograd isolate: Vrequiresgrad butcompatibilitydetached; noCNN gradient.
  k.zero_grad();v=values.clone().requires_grad_();s=compatibility(v);assert not s.requires_grad
  bias=k.scr_beta[None,:,None]*s[:,None,:];(bias*torch.randn(bias.shape,generator=gen)).sum().backward();assert v.grad is None;assert all(p.grad is None for p in k.conv2d_features.parameters());assert all(p.grad is None for p in k.conv3d_features.parameters());assert all(p.grad is None for p in k.relation_scores.parameters());assert k.scr_beta.grad.abs().sum()>0
  c.eval();k.eval();assert float((c(x,t)[1]-k(x,t)[1]).abs().max())<1e-6
  for h in hooks:h.remove()
  rows.append({'seed':seed,'original_initial_hash':o,'beta0_weights_error':werr,'beta0_tokens_error':terr,'beta0_logits_error_train':err,'BN_Transformer_calls_train':{n:1 for n in counts},'beta_gradient':beta_grad,'isolated_semantic_bias_has_no_V_CNN_scorer_gradient':True,'center_row_col':[6,6],'V_shape':list(values.shape),'s_sem_shape':list(sem.shape),'s_spec_final_shape':list(wk.shape),'main_RNG_draws_unchanged':True})
 dump(OUT/'verification.json',{'passed':True,'parameters_added':4,'parameters_K1':378015,'beta_initial':[0.,0.,0.,0.],'seeds':SEEDS,'checks':rows});print('PASS: beta0 W/T/logits,4paramsonly,actualVcenter,detachedsemanticbias,beta gradient,RNG/BN/blocks')
if __name__=='__main__':main()
