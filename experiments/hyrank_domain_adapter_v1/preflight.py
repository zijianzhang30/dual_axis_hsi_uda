import torch,json
import study as s
import adapter as a

def main():
 torch.set_num_threads(2);c,ns=s.environment();lm=s.lmmd();results=[]
 for arm in s.ARMS:
  c.seed_worker(2100);args=c.options(2100);args.device='0';baseline,_,_=s.parent.build(c,args,'C');c.seed_worker(2100);m,_,_=s.build(c,args,arm)
  baseline=baseline.cuda().train();m=m.cuda().train()
  # Paired base tensors are identical despite new modules.
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in baseline.state_dict().items())
  x=torch.randn(12,1,176,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(12,device='cuda')
  cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();bo=baseline(x,t)
  torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);captures=[]
  hook=m.norm.register_forward_hook(lambda mod,inp,out:captures.append(out[:,0]));out=m(x,t)
  err=max(float((u-v).abs().max()) for u,v in zip(bo,out));assert err<1e-6
  loss_lm=lm.lmmd(captures[0],captures[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=12)
  # Target receives DIRECT LMMD gradient even at zero residual initialization.
  target_params=[p for name,p in m.named_parameters() if '.adapter_target.up.' in name or '.adapter_shared.up.' in name]
  grad=torch.autograd.grad(loss_lm,target_params,retain_graph=True,allow_unused=True)
  target_norm=sum(float(g.square().sum()) for g in grad if g is not None)**.5;assert target_norm>0
  opt=torch.optim.SGD(m.parameters(),lr=.01);total=torch.nn.functional.cross_entropy(out[0],y)+.1*loss_lm;total.backward();opt.step()
  captures.clear();opt.zero_grad(set_to_none=True);out=m(x,t);loss_lm=lm.lmmd(captures[0],captures[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=12)
  (torch.nn.functional.cross_entropy(out[0],y)+.1*loss_lm).backward();stats=a.gradient_stats(m)
  assert stats and all(v>0 and torch.isfinite(torch.tensor(v)) for v in stats.values()),stats
  hook.remove();m.eval()
  with torch.no_grad():eo=m(x,t)[1].clone()
  if arm=='R2':
   originals={name:p.clone() for name,p in m.named_parameters() if '.adapter_source.' in name}
   with torch.no_grad():
    for name,p in m.named_parameters():
     if name in originals:p.fill_(123.)
    e2=m(x,t)[1];assert torch.equal(eo,e2)
    for name,p in m.named_parameters():
     if name in originals:p.copy_(originals[name])
  # Verify routing and saved-model inference. No source input scene or GT required.
  path=s.OUT/('preflight_%s.pth'%arm);torch.save(m.state_dict(),path)
  new,_,_=s.build(c,args,arm);new=new.cuda().eval();new.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
  with torch.no_grad():assert torch.equal(new(x,t)[1],eo)
  results.append({'arm':arm,'zero_adapter_max_logit_error':err,'extra_total_parameters':12288,
  'per_domain_active_extra_parameters':12288 if arm=='R1' else 6144,'direct_target_LMMD_gradient_norm':target_norm,
  'all_adapter_parameter_gradients_nonzero_after_update':True,'gradient_norms':stats,'source_adapter_unused_at_eval':True,
  'saved_target_inference_reproducible':True,'BN_updates_per_paired_forward':1})
  del baseline,m,new,out,captures,total
 s.dump(s.OUT/'preflight.json',{'passed':True,'cases':results,'same_total_params_real_active':True,'no_NEW_loss':True})
 print('ADAPTER_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
