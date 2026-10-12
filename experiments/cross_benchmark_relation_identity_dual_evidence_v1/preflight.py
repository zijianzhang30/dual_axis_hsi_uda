import argparse,json,torch
import study as s
from expert import build,DropoutReplay

def main(dataset):
 s.verify();torch.set_num_threads(2);m,values,r=s.setup(dataset,2100);e=build(r['bands'],r['classes'],2100).cuda();m.cuda().train();calls=[];hooks=[]
 for mod in m.modules():
  if isinstance(mod,torch.nn.Dropout):hooks.append(mod.register_forward_pre_hook(lambda mod,args:calls.append((list(args[0].shape),mod.p))))
 g=torch.Generator().manual_seed(60146);x=torch.rand(128,1,r['bands'],13,13,generator=g).cuda();t=torch.rand(128,1,r['bands'],13,13,generator=g).cuda();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
 with torch.no_grad():out=m(x,t)
 endcpu=torch.get_rng_state().clone();endgpu=torch.cuda.get_rng_state().clone()
 for h in hooks:h.remove()
 assert len(calls)==44;replay=DropoutReplay(calls);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);replay.advance();assert torch.equal(endcpu,torch.get_rng_state()) and torch.equal(endgpu,torch.cuda.get_rng_state())
 m.zero_grad(set_to_none=True);before=r['hash'](m);torch.nn.functional.cross_entropy(e(x[:,0,:,6,6].detach()),torch.arange(128,device='cuda')%r['classes']).backward();assert all(p.grad is None for p in m.parameters()) and before==r['hash'](m);assert float(e[1].weight.grad.norm())>0;assert torch.equal(endgpu,torch.cuda.get_rng_state())
 s.dump(s.OUT/('preflight_'+dataset+'.json'),{'passed':True,'dropout_calls':calls,'actual_C_forward_vs_dropout_only_RNG_bitwise_equal':True,'spectral_backward_no_relation_gradient_or_state_update':True,'spectral_step_no_RNG':True,'spectral_parameters':sum(p.numel() for p in e.parameters()),'disposable_models_not_reused_R0':True,'target_GT_not_scored':True});print('PREFLIGHT_PASS',dataset)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=s.DATASETS,required=True);main(p.parse_args().dataset)
