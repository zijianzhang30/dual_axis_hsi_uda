import argparse,hashlib,json
import numpy as np,torch
import study as s
def main(d):
 s.verify();torch.set_num_threads(2);rows=[];shared=None;dropout=None
 for r in s.RELATIONS:
  m,values,rt=s.setup(d,r,2100);params={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith(('conv_a.','relation_scores.'))}
  if shared is None:shared=params
  else:assert set(shared)==set(params) and all(torch.equal(shared[k],params[k]) for k in params)
  m.cuda().train();calls=[];hs=[z.register_forward_pre_hook(lambda mod,args:calls.append((list(args[0].shape),mod.p))) for z in m.modules() if isinstance(z,torch.nn.Dropout)];g=torch.Generator().manual_seed(732105);x=torch.rand(128,1,rt['bands'],13,13,generator=g).cuda();t=torch.rand(128,1,rt['bands'],13,13,generator=g).cuda();cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();z=m(x,t)[0];endcpu=torch.get_rng_state().clone();endcuda=torch.cuda.get_rng_state().clone();assert len(calls)==44
  if dropout is None:dropout=calls
  else:assert dropout==calls
  # Native and Joint paths have the same only-stochastic operators.
  sys=s.module('factorial_dropout_replay',s.ROOT/'experiments/cross_benchmark_relation_identity_dual_evidence_v1/expert.py');replay=sys.DropoutReplay(calls);torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);replay.advance();assert torch.equal(endcpu,torch.get_rng_state()) and torch.equal(endcuda,torch.cuda.get_rng_state());loss=torch.nn.functional.cross_entropy(z,torch.arange(128,device='cuda')%rt['classes']);loss.backward();assert float(m.nn1.weight.grad.norm())>0
  bn=[int(v.num_batches_tracked) for v in m.modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))];assert all(v==(1 if r.startswith('J') else 2) for v in bn);rows.append({'relation':r,'initial_shared_tensors_identical':True,'dropout44_CPU_CUDA_RNG_bitwise_validated':True,'BN_counts':bn,'source_CE_classifier_gradient_nonzero':True,'parameters':sum(p.numel() for p in m.parameters())});print('PREFLIGHT_ARM_PASS',d,r,flush=True)
  for h in hs:h.remove()
  del m,values,rt,z,loss,replay,x,t;torch.cuda.empty_cache()
 s.dump(s.OUT/('preflight_'+d+'.json'),{'passed':True,'dataset':d,'rows':rows,'target_GT_not_scored':True,'disposable_no_formal_training':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=s.DATASETS,required=True);main(p.parse_args().dataset)
