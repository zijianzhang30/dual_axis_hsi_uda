import torch
from common import OUT,SEEDS,environment,build,state_hash,dump

def main():
 torch.set_num_threads(2);device=torch.device('cuda:0');env=environment();rows=[]
 for seed in SEEDS:
  models={};originals={}
  for arm in ['C','V0','V1']:
   env[4](seed);models[arm],originals[arm]=build(seed,arm,'houston')
  c,v0,v1=[models[a] for a in ['C','V0','V1']];assert len(set(originals.values()))==1
  assert state_hash(v0)==state_hash(v1);initial_full=state_hash(v1)
  for model in [v0,v1]:
   assert sum(p.numel() for p in model.parameters())==382204
   assert sum(p.numel() for p in model.value_calibration.parameters())==4193
   assert all(torch.equal(value,model.state_dict()[name]) for name,value in c.state_dict().items())
   added=set(dict(model.named_parameters()))-set(dict(c.named_parameters()));assert added=={'value_calibration.gamma','value_calibration.net.0.weight','value_calibration.net.0.bias','value_calibration.net.2.weight','value_calibration.net.2.bias'}
  for model in models.values():model.to(device)
  gen=torch.Generator().manual_seed(92000);x=torch.randn((4,1,48,13,13),generator=gen).to(device);t=torch.randn(x.shape,generator=gen).to(device);tc=t.clone().requires_grad_();tv=t.clone().requires_grad_();captures={};hooks=[];counts={}
  for arm,model in [('C',c),('V1',v1)]:
   orig=model._forward_semantic_tokens;captures[arm]=[]
   def cap(features,arm=arm,model=model,orig=orig):
    w=model.relation_scores(model._relation_pending[0]);tok=orig(features);captures[arm].append((w.detach(),tok.detach(),features.detach()));return tok
   model._forward_semantic_tokens=cap
  for name,module in v1.named_modules():
   if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) or module.__class__.__name__.startswith('Block'):
    counts[name]=0
    def hook(m,a,b,name=name):counts[name]+=1
    hooks.append(module.register_forward_hook(hook))
  c.train();v1.train();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state();oc=c(x,tc);aftercpu=torch.get_rng_state();aftercuda=torch.cuda.get_rng_state();torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);ov=v1(x,tv)
  assert torch.equal(aftercpu,torch.get_rng_state()) and torch.equal(aftercuda,torch.cuda.get_rng_state())
  err=max(float((a-b).abs().max()) for a,b in zip(oc,ov));werr=max(float((a[0]-b[0]).abs().max()) for a,b in zip(captures['C'],captures['V1']));terr=max(float((a[1]-b[1]).abs().max()) for a,b in zip(captures['C'],captures['V1']));assert max(err,werr,terr)<1e-6
  assert all(n==1 for n in counts.values()),counts
  feature=captures['V1'][0][2];d=v1.value_calibration.normalized_relation(feature);assert d[:,:,6,6].abs().max()==0 and torch.isfinite(d).all();assert torch.equal(d,torch.nn.functional.normalize(feature-feature[:,:,6:7,6:7],dim=1,eps=1e-6))
  y=torch.arange(4,device=device);torch.nn.functional.cross_entropy(oc[0],y).backward();torch.nn.functional.cross_entropy(ov[0],y).backward()
  gamma_grad=float(v1.value_calibration.gamma.grad);assert abs(gamma_grad)>0 and torch.isfinite(v1.value_calibration.gamma.grad)
  assert all(p.grad is not None and p.grad.abs().max()==0 for p in v1.value_calibration.net.parameters())
  shared_grad_error=max(float((p.grad-dict(v1.named_parameters())[n].grad).abs().max()) for n,p in c.named_parameters() if p.grad is not None);assert shared_grad_error<1e-6,shared_grad_error
  assert tv.grad is not None and tv.grad.norm()>0
  opt=torch.optim.SGD(v1.parameters(),lr=.01);before_MLP=[p.detach().clone() for p in v1.value_calibration.net.parameters()];opt.step();assert v1.value_calibration.gamma!=0;assert all(torch.equal(a,b) for a,b in zip(before_MLP,v1.value_calibration.net.parameters()))
  v1.zero_grad();torch.nn.functional.cross_entropy(v1(x,t)[0],y).backward();mlp_gradient=float(sum(p.grad.norm() for p in v1.value_calibration.net.parameters()));assert mlp_gradient>0 and all(torch.isfinite(p.grad).all() for p in v1.parameters() if p.grad is not None)
  for h in hooks:h.remove()
  # Isolate relation-to-value backprop, including center reference, without classifier identity path.
  v1.zero_grad();toy=torch.rand((2,64,13,13),generator=gen).to(device).requires_grad_();dt=v1.value_calibration.normalized_relation(toy);branch=v1.value_calibration.net(dt.permute(0,2,3,1));coefficient=torch.randn(branch.shape,generator=gen).to(device);(branch*coefficient).sum().backward();assert toy.grad is not None and toy.grad.norm()>0 and toy.grad[:,:,6,6].norm()>0;center_grad=float(toy.grad[:,:,6,6].norm())
  # Restore common state, reset scalar, audit running-stat eval equivalence.
  v1.load_state_dict(c.state_dict(),strict=False);v1.value_calibration.gamma.data.zero_();c.eval();v1.eval();v0.eval();evalerr=float((c(x,t)[1]-v1(x,t)[1]).abs().max());assert evalerr<1e-6
  rows.append({'seed':seed,'original_initial_sha256':originals['C'],'V0_V1_initial_full_sha256':initial_full,'gamma0_weights_error':werr,'gamma0_tokens_error':terr,'gamma0_train_logits_error':err,'gamma0_eval_logits_error':evalerr,'shared_parameter_gradient_error_gamma0':shared_grad_error,'BN_and_Transformer_calls_per_train_pair':{n:1 for n in counts},'gamma_initial_gradient':gamma_grad,'MLP_first_step_gradient_zero_expected':True,'MLP_gradient_after_gamma_update':mlp_gradient,'center_relation_gradient_isolated':center_grad,'target_via_JointBN_gradient_norm':float(tv.grad.norm()),'original_CPU_CUDA_rng_preserved':True,'norm_axis':'channel dim1 L2 eps1e-6','center':[6,6],'value_shape':list(feature.shape)})
  del models,c,v0,v1;torch.cuda.empty_cache()
 dump(OUT/'verification.json',{'passed':True,'seeds':list(SEEDS),'parameters_added':4193,'V1_V0_parameters':382204,'gamma_initial':0.,'checks':rows});print('PASS: gamma0 W/T/logits and shared gradients,RNG/BN/Transformer,center/target paths,delayed MLP gradient,capacity matched')
if __name__=='__main__':main()
