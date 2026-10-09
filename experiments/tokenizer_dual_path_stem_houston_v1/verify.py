import torch
from common import OUT,SEEDS,environment,build,state_hash,dump

def main():
 torch.set_num_threads(2);env=environment();rows=[];device=torch.device('cuda:0')
 for seed in SEEDS:
  models={};originals={}
  for arm in ['C','N0','N1']:
   env[4](seed);models[arm],originals[arm]=build(seed,arm,'houston')
  c,n0,n1=[models[a] for a in ['C','N0','N1']];assert len(set(originals.values()))==1
  assert state_hash(n0)==state_hash(n1)
  for m in [n0,n1]:
   assert sum(p.numel() for p in m.parameters())==378011
   stem=m.conv3d_features[0];assert sum(p.numel() for p in stem.parameters())==224
   assert torch.equal(torch.cat([stem.absolute.weight,stem.relative.weight]),c.conv3d_features[0].weight)
   assert torch.equal(torch.cat([stem.absolute.bias,stem.relative.bias]),c.conv3d_features[0].bias)
   assert all(torch.equal(v,m.state_dict()[k]) for k,v in c.state_dict().items() if not k.startswith('conv3d_features.0.'))
  initial_hash=state_hash(n0)
  for m in models.values():m.to(device)
  gen=torch.Generator().manual_seed(92000);x=torch.randn((4,1,48,13,13),generator=gen).to(device);t=torch.randn(x.shape,generator=gen).to(device).requires_grad_()
  relation=n1.conv3d_features[0].relation(t);rerr=float((relation[:,0]-n1.relation_scores.relations(t)).abs().max());assert rerr<1e-6
  assert relation[:,:,:,6,6].abs().max()==0 and torch.isfinite(relation).all()
  # Old full path vs new split X/X control; record rounding from separate CUDA kernels.
  c.train();n0.train();n1.train();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state();oc=c(x,t);aftercpu=torch.get_rng_state();aftercuda=torch.cuda.get_rng_state()
  torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);o0=n0(x,t)
  assert torch.equal(aftercpu,torch.get_rng_state()) and torch.equal(aftercuda,torch.cuda.get_rng_state())
  control_err=max(float((a-b).abs().max()) for a,b in zip(oc,o0));assert control_err<.002,control_err
  counts={};hooks=[]
  for name,m in n1.named_modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or m.__class__.__name__.startswith('Block'):
    counts[name]=0
    def hook(m,a,b,name=name):counts[name]+=1
    hooks.append(m.register_forward_hook(hook))
  torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);o1=n1(x,t)
  assert torch.equal(aftercpu,torch.get_rng_state()) and torch.equal(aftercuda,torch.cuda.get_rng_state())
  assert all(v==1 for v in counts.values()),counts
  torch.nn.functional.cross_entropy(o1[0],torch.arange(4,device=device)).backward()
  gradients={name:float(module.weight.grad.norm()) for name,module in [('absolute',n1.conv3d_features[0].absolute),('relative',n1.conv3d_features[0].relative),('Conv2D',n1.conv2d_features[0])]}
  gradients['relation_scorer']=float(sum(p.grad.norm() for p in n1.relation_scores.parameters()))
  assert all(v>0 for v in gradients.values());assert t.grad is not None and t.grad.abs().sum()>0
  before={k:v.detach().clone() for k,v in n1.conv3d_features[0].named_parameters()};torch.optim.SGD(n1.parameters(),lr=.01).step();assert all(not torch.equal(v,dict(n1.conv3d_features[0].named_parameters())[k]) for k,v in before.items() if k.endswith('weight'))
  for h in hooks:h.remove()
  # Checkpoint reconstruction exactly reproduces target-only image input, native evaluation pairing.
  env[4](seed);clone,_=build(seed,'N1','houston');clone.load_state_dict(n1.state_dict());clone.to(device).eval();n1.eval()
  with torch.inference_mode():reloaderr=float((clone(x,x)[1]-n1(x,x)[1]).abs().max());assert reloaderr==0
  rows.append({'seed':seed,'original_initial_sha256':originals['C'],'N0_N1_initial_sha256':initial_hash,'copied_original_filters_exact':True,'relation_matches_C_max_error':rerr,'center':[6,6],'N0_original_C_train_logits_max_error_GPU':control_err,'control_note':'split convolution may change floating-point order; N0 trained as explicit matched control rather than assumed C-equivalent trajectory','BN_Transformer_calls_per_train_pair':counts,'gradient_norms':gradients,'target_through_joint_BN_gradient_norm':float(t.grad.norm()),'optimizer_updates_both_paths':True,'CPU_CUDA_RNG_streams_unchanged':True,'reload_target_input_error':reloaderr})
  del models,c,n0,n1,clone;torch.cuda.empty_cache()
 dump(OUT/'verification.json',{'passed':True,'seeds':list(SEEDS),'parameters_all':378011,'stem_parameters_all':224,'extra_parameters':0,'checks':rows});print('PASS: matched capacity/initial filters, actual relation, gradients/update,BN/Transformer/RNG paths,reload')
if __name__=='__main__':main()
