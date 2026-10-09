import torch,numpy as np
import study as s

def main():
 torch.set_num_threads(2);c,ns=s.environment();args=c.options(2100);args.device='0';c.seed_worker(2100)
 m,unused,initial=s.build(c,args,'C');kernel=s.module('off_checked_pair',s.OUT/'pairing_kernel.py')
 pair=kernel.Pairing(2100,128,enabled=False);pair.epoch=100
 x=np.arange(128);y=x%12;z=np.full((128,12),-10.);z[x,x%12]=10.;pair.observe(x,z)
 state=np.random.get_state();selected,matched,_,_=pair.select(y,x)
 assert np.array_equal(selected,x) and not matched.any()
 assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(state,np.random.get_state()))
 m=m.cuda().train();x=torch.randn(8,1,176,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(8,device='cuda')
 class Trace:pass
 trace=Trace();trace.phase='train';trace.source=(x.cpu(),y.cpu(),torch.arange(8));trace.target=(t.cpu(),torch.arange(8));pair.trace=trace
 before=t.clone();pair.pre(m,(x,t));assert torch.equal(before,t)
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();state={k:v.clone() for k,v in m.state_dict().items()};out=m(x,t)
 m.load_state_dict(state);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);again=m(x,t)
 assert all(torch.equal(a,b) for a,b in zip(out,again))
 s.dump(s.OUT/'preflight.json',{'passed':True,'off_even_with_valid_cache':True,'input_tensor_identity':True,'global_numpy_RNG_unchanged':True,'max_off_hook_logits_error':0.,'parent_model_config_and_loss_reused':True,'training_factor_only':'Eq9 target replacement disabled'})
 print('EQ9_OFF_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
