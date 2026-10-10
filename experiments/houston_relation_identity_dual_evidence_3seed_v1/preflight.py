import copy,json,torch
import torch.nn.functional as F
import study as s
from expert import build

def main():
 s.verify();torch.set_num_threads(2);c=s.environment();a,_,_,_,_=c.setup('houston','C',2100);b=copy.deepcopy(a);e=build().cuda();a.cuda();b.cuda();opts=[torch.optim.SGD(v.parameters(),lr=.01) for v in (a,b,e)];assert not {id(p) for p in b.parameters()}&{id(p) for p in e.parameters()}
 g=torch.Generator().manual_seed(382100);x=torch.rand(8,1,48,13,13,generator=g).cuda();t=torch.rand(8,1,48,13,13,generator=g).cuda();y=(torch.arange(8)%7).cuda();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();states=[]
 for i,m in enumerate((a,b)):
  torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);m.train();opts[i].zero_grad(set_to_none=True);z=m(x,t);F.cross_entropy(z[0],y).backward();grad={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()};opts[i].step()
  if i==1:
   before=c.state_hash(m);opts[2].zero_grad(set_to_none=True);F.cross_entropy(e(x[:,0,:,6,6].detach()),y).backward();assert all((v is None and dict(m.named_parameters())[n].grad is None) or (v is not None and torch.equal(v,dict(m.named_parameters())[n].grad)) for n,v in grad.items());opts[2].step();assert before==c.state_hash(m)
  states.append({'logits':[v.detach().clone() for v in z],'grad':grad,'cpu':torch.get_rng_state().clone(),'gpu':torch.cuda.get_rng_state().clone(),'state':c.state_hash(m)})
 assert all(torch.equal(x,y) for x,y in zip(states[0]['logits'],states[1]['logits'])) and states[0]['state']==states[1]['state'];assert torch.equal(states[0]['cpu'],states[1]['cpu']) and torch.equal(states[0]['gpu'],states[1]['gpu']);assert all((v is None and states[1]['grad'][n] is None) or (v is not None and torch.equal(v,states[1]['grad'][n])) for n,v in states[0]['grad'].items());assert float(e[1].weight.grad.norm())>0
 s.dump(s.OUT/'preflight.json',{'passed':True,'original_logits_and_gradients_and_first_update_bitwise_equal':True,'spectral_backward_does_not_change_relation_gradients':True,'spectral_update_does_not_change_relation_state_or_BN':True,'RNG_bitwise_equal':True,'disjoint_parameter_optimizers':True,'spectral_gradient_norm':float(e[1].weight.grad.norm()),'disposable_models_only':True});print('PREFLIGHT_PASS')
if __name__=='__main__':main()
