import copy,json,torch
import torch.nn.functional as F
import study as s
import adapter as a

def main():
 torch.set_num_threads(2);s.verify();checks=[]
 for seed in s.SEEDS:
  for arm in s.ARMS:
   c=s.environment();m,loaders,helper,classes,audit=c.setup('houston','C',seed);before=torch.get_rng_state().clone();a.install(m,'R1',seed);assert torch.equal(before,torch.get_rng_state());m=m.cuda().train()
   x=torch.randn(14,1,48,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(14,device='cuda')%7
   cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
   # Identical baseline tensors; same seed/input/state/dropout for zero-initialized adapter.
   ref,_,_,_,_=c.setup('houston','C',seed);ref=ref.cuda().train();ref.load_state_dict({k:v for k,v in m.state_dict().items() if '.adapter_' not in k})
   torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);base=ref(x,t);endcpu=torch.get_rng_state().clone();endgpu=torch.cuda.get_rng_state().clone()
   torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);capt=[];hook=m.norm.register_forward_hook(lambda mod,inp,out:capt.append(out[:,0]));out=m(x,t)
   err=max(float((aa-bb).abs().max()) for aa,bb in zip(base,out));assert err<1e-6;assert torch.equal(endcpu,torch.get_rng_state()) and torch.equal(endgpu,torch.cuda.get_rng_state())
   assert len(capt)==4 and m.relation_scores(x).shape==(14,4,169)
   assert float(m.relation_scores.relations(x)[:,:,6,6].abs().max())==0
   ce=F.cross_entropy(out[0],y);lm=c.lmmd().lmmd(capt[0],capt[1],y,out[1].softmax(1),CLASS_NUM=7,BATCH_SIZE=14) if arm=='LMMD1' else torch.zeros((),device='cuda')
   assert torch.isfinite(ce+lm).all();(ce+.1*lm).backward();g=a.gradients(m)
   assert all(v>0 for k,v in g.items() if '.up.' in k);assert all(v==0 for k,v in g.items() if '.down.' in k)
   torch.optim.SGD(m.parameters(),lr=.01).step();m.zero_grad(set_to_none=True);capt.clear();out=m(x,t);ce=F.cross_entropy(out[0],y)
   lm=c.lmmd().lmmd(capt[0],capt[1],y,out[1].softmax(1),CLASS_NUM=7,BATCH_SIZE=14) if arm=='LMMD1' else torch.zeros((),device='cuda')
   (ce+.1*lm).backward();assert all(v>0 for v in a.gradients(m).values())
   checks.append({'seed':seed,'arm':arm,'original_initial_hash':audit,'zero_adapter_logits_error':err,'CPU_CUDA_rng_preserved':True,'first_step_up_gradient_nonzero_down_zero_expected':True,'after_first_update_all_adapter_gradients_nonzero':True,'center_anchor13':True,'extra_parameters':6144,'no_target_labels_in_preflight':True})
   hook.remove();del m,ref,loaders,capt,base,out;torch.cuda.empty_cache()
 s.dump(s.OUT/'preflight.json',{'passed':True,'checks':checks,'new_training_limit':6,'HyRANK_reused':True,'Pavia_deferred':True});print('HOUSTON_SHARED_ADAPTER_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
