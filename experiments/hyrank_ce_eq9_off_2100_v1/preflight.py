import json
import torch,numpy as np
import torch.nn.functional as F
import study as s

def main():
 torch.set_num_threads(2);c,ns=s.environment();args=c.options(2100);args.device='0';c.seed_worker(2100)
 m,unused,initial=s.build(c,args,'C')
 expected=json.loads((s.CONTROL/'C_2100/config.json').read_text())
 assert c.helper.state_hash(m)==expected['initial_student']
 kernel=s.module('ce_off_checked_pair',s.OUT/'pairing_kernel.py');pair=kernel.Pairing(2100,128,enabled=False);pair.epoch=100
 ids=np.arange(128);ys=ids%12;probs=np.full((128,12),-10.);probs[ids,ys]=10.;pair.observe(ids,probs)
 state=np.random.get_state();selected,matched,_,_=pair.select(ys,ids)
 assert np.array_equal(selected,ids) and not matched.any()
 assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(state,np.random.get_state()))
 m=m.cuda().train();x=torch.randn(12,1,176,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(12,device='cuda')
 class Trace:pass
 trace=Trace();trace.phase='train';trace.source=(x.cpu(),y.cpu(),torch.arange(12));trace.target=(t.cpu(),torch.arange(12));pair.trace=trace
 before=t.clone();pair.pre(m,(x,t));assert torch.equal(before,t)
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();initial_state={k:v.clone() for k,v in m.state_dict().items()}
 capture=[];hook=m.norm.register_forward_hook(lambda module,inputs,output:capture.append(output[:,0]))
 out=m(x,t);ce=F.cross_entropy(out[0],y)
 # Preflight only: prove removing a zero-weight LMMD has the same CE gradients.
 lm=s.lmmd();lmd=lm.lmmd(capture[0],capture[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=12)
 assert torch.isfinite(lmd).all();(ce+0.*lmd).backward()
 grad_ref={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
 cpu_end=torch.get_rng_state().clone();gpu_end=torch.cuda.get_rng_state().clone()
 m.load_state_dict(initial_state);m.zero_grad(set_to_none=True);capture.clear();torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
 again=m(x,t);F.cross_entropy(again[0],y).backward()
 max_logits=max(float((a-b).abs().max()) for a,b in zip(out,again));assert max_logits<1e-6
 grad_error=max(float((p.grad-grad_ref[k]).abs().max()) for k,p in m.named_parameters() if k in grad_ref)
 assert grad_error<1e-6
 assert torch.equal(cpu_end,torch.get_rng_state()) and torch.equal(gpu_end,torch.cuda.get_rng_state())
 bn={k:int(v.num_batches_tracked) for k,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==1 for v in bn.values())
 scorer_grads={k:float(p.grad.norm()) for k,p in m.named_parameters() if 'relation' in k and p.grad is not None};assert scorer_grads and any(v>0 for v in scorer_grads.values())
 hook.remove()
 s.dump(s.OUT/'preflight.json',{'passed':True,'initial_hash_matches_CE_LMMD_Eq9_off':True,'off_even_with_valid_cache':True,'input_tensor_identity':True,'global_numpy_RNG_unchanged':True,'zero_LMMD_logits_max_error':max_logits,'zero_LMMD_gradient_max_error':grad_error,'CPU_CUDA_rng_match':True,'BN_calls':bn,'relation_scorer_gradient_norms':scorer_grads,'training_objective':'source CE ONLY; no LMMD called during training','target_labels_not_accessed':True})
 print('CE_ONLY_EQ9_OFF_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
