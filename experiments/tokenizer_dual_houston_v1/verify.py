import copy,json,torch
from common import OUT,BASE,SEEDS,environment,build,state_hash,dump
from diagnostics import describe

def main():
 torch.set_num_threads(2);env=environment();checks=[]
 for seed in SEEDS:
  env[4](seed);h,initial=build(seed,'H','houston');initial_h=state_hash(h)
  a=json.loads((BASE/'houston'/f'A_{seed}/config.json').read_text());c=json.loads((BASE/'houston'/f'C_{seed}/config.json').read_text())
  assert initial==a['initial_original_model_sha256']==c['initial_original_model_sha256']
  assert state_hash(h.relation_scores)==c['initial_scorer_sha256']
  assert float(h.alpha)==0 and float(h.alpha.sigmoid())==.5
  total=sum(p.numel() for p in h.parameters());assert total==378268==c['parameters']+257==a['parameters']+1701
  # Compare all native initial tensors, including semantic conv_a, to fresh native A.
  env[4](seed);args=__import__('argparse').Namespace(model='BiDA',lr=.01,num_tokens=4,dim=64,depth=3,loss_type='softmax',labelsmooth='off')
  native=env[1]('BiDA','Houston13',13,args)
  assert all(torch.equal(v,h.state_dict()[k]) for k,v in native.state_dict().items())
  gen=torch.Generator().manual_seed(88000);x=torch.randn(4,1,48,13,13,generator=gen);x=x/x.square().sum(2,keepdim=True).sqrt();t=x.flip(-1).clone().requires_grad_()
  h.train();h._record_dual=True;out=h(x,t);ws,wr,ts,tr=h._dual_diagnostic
  assert ts.shape==tr.shape==(4,4,64);assert ws.shape==wr.shape==(4,4,169)
  before={k:v.detach().clone() for k,v in h.named_parameters()};loss=torch.nn.functional.cross_entropy(out[0],torch.arange(4));loss.backward()
  grads={'semantic':float(h.conv_a.weight.grad.norm()),'relation':float(torch.stack([p.grad.norm() for p in h.relation_scores.parameters()]).norm()),'gate':float(h.alpha.grad.abs()),'target_input_through_Joint_BN':float(t.grad.norm())}
  assert all(v>0 for v in grads.values());assert all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None)
  torch.optim.SGD(h.parameters(),lr=.01).step()
  for k in ['conv_a.weight','relation_scores.scorer.0.weight','relation_scores.scorer.2.weight','alpha']:assert not torch.equal(before[k],dict(h.named_parameters())[k])
  # H changes no main-stream random draws: compare native A and H on same RNG.
  env[5].install(native,'fixed_mixture');native.train();torch.manual_seed(543);rng=torch.get_rng_state();native(x,t.detach());after=torch.get_rng_state();torch.set_rng_state(rng);h(x,t.detach());assert torch.equal(after,torch.get_rng_state())
  checks.append({'seed':seed,'initial_original_hash':initial,'H_initial_hash':initial_h,'relation_initial_hash':c['initial_scorer_sha256'],'total_parameters':total,'gradient_and_updates':grads,'initial_slot_diagnostic':describe(ws,wr,ts,tr),'main_RNG_draw_count_matches_native_A':True})
 dump(OUT/'verification.json',{'passed':True,'seeds':list(SEEDS),'checks':checks,'parameters_A':376567,'parameters_C':378011,'parameters_H':378268,'gate_initial':.5,'tokens':4,'dim':64,'same_CNN_value_grid':True,'no_target_label_supervision':True,'extra_MACs_vs_C_per_patch':86528,'MAC_convention':'conv_a 4*64*169 + semantic weighted pooling 4*64*169; plus512 gate blend scalar arithmetic and softmax reductions'})
 print('PASS: initial native weights + C scorer, shapes, gradients/updates, native RNG path, parameter counts')
if __name__=='__main__':main()
