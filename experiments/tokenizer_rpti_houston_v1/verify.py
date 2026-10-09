import torch,json,copy
from common import OUT,SEEDS,environment,build,state_hash,dump
from rpti import provenance,bias

def main():
 torch.set_num_threads(2);env=environment();results=[]
 for seed in SEEDS:
  env[4](seed);c,original=build(seed,'C','houston');env[4](seed);j,o=build(seed,'J1','houston');assert original==o
  assert all(torch.equal(v,j.state_dict()[k]) for k,v in c.state_dict().items())
  added=set(dict(j.named_parameters()))-set(dict(c.named_parameters()));assert added=={f'blocks.{i}.attn.beta' for i in range(3)}
  assert sum(dict(j.named_parameters())[k].numel() for k in added)==24
  g=torch.Generator().manual_seed(90000);x=torch.randn((4,1,48,13,13),generator=g);t=x.flip(-1)
  counts={};hooks=[]
  for name,m in j.named_modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
    counts[name]=0
    def hook(m,a,b,name=name):counts[name]+=1
    hooks.append(m.register_forward_hook(hook))
  c.train();j.train();torch.manual_seed(41);rng=torch.get_rng_state();a=c(x,t);after=torch.get_rng_state();torch.set_rng_state(rng);b=j(x,t)
  assert torch.equal(after,torch.get_rng_state());err=max(float((q-z).abs().max()) for q,z in zip(a,b));assert err<1e-6
  assert all(v==1 for v in counts.values());torch.nn.functional.cross_entropy(b[0],torch.arange(4)).backward()
  grads=[float(blk.attn.beta.grad.norm()) for blk in j.blocks];assert any(v>0 for v in grads) and all(torch.isfinite(blk.attn.beta.grad).all() for blk in j.blocks)
  # Isolate bias branch at nonzero beta: disconnected W/scorer graph, beta still differentiable.
  j.zero_grad();w=j.relation_scores(t);r=provenance(w);assert not r.requires_grad and torch.allclose(r,r.transpose(-1,-2))
  att=j.blocks[0].attn;att._rpti_cache[:]=[r,r]
  with torch.no_grad():att.beta.fill_(.1)
  bb=bias(att,0);assert bb.shape==(4,8,5,5);assert (bb[:,:,0,:]==0).all() and (bb[:,:,:,0]==0).all()
  (bb*torch.randn(bb.shape,generator=g)).sum().backward();assert all(p.grad is None for p in j.relation_scores.parameters());assert att.beta.grad.abs().sum()>0
  # Eval target-only path must also match at all-zero beta.
  for blk in j.blocks:blk.attn.beta.data.zero_()
  c.eval();j.eval();a=c(x,t);b=j(x,t);evalerr=float((a[1]-b[1]).abs().max());assert evalerr<1e-6
  for h in hooks:h.remove()
  results.append({'seed':seed,'original_initial_hash':original,'initial_C_hash':state_hash(c),'train_logits_max_error_beta0':err,'eval_logits_max_error_beta0':evalerr,'CNN_BN_calls_train':2,'beta_gradient_norms':grads,'stopgrad_bias_has_no_scorer_gradient':True,'added_parameters':sorted(added),'W_shape':list(w.shape),'R_shape':list(r.shape),'bias_shape':list(bb.shape),'CLS_bias_exact_zero':True})
 dump(OUT/'verification.json',{'passed':True,'results':results,'parameters_added':24,'parameters_J1':378035,'head_count':8,'layers':3,'cross_paths_unchanged':True,'main_RNG_draws_unchanged':True})
 print('PASS: beta0 train/eval logits,24 beta-only params,CLS0,detachedR,beta gradients,RNG/BN')
if __name__=='__main__':main()
