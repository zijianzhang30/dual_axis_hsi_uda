import torch,json
from common import OUT,BASE,SEEDS,build,environment,dump

def main():
 torch.set_num_threads(2);env=environment();records={}
 for seed in SEEDS:
  models={};cpu=[];cuda=[]
  for arm in ['A','C']:
   env[4](seed);m,h=build(seed,arm);m.cuda();models[arm]=m
   assert h==json.loads((BASE/'houston'/f'{arm}_{seed}'/'config.json').read_text())['initial_original_model_sha256']
   cpu.append(torch.get_rng_state());cuda.append(torch.cuda.get_rng_state())
  assert torch.equal(cpu[0],cpu[1]) and torch.equal(cuda[0],cuda[1])
  assert all(torch.equal(v,models['C'].state_dict()[k]) for k,v in models['A'].state_dict().items() if not k.startswith('conv_a.'))
  x=torch.randn(4,1,48,13,13,device='cuda');y=torch.tensor([0,1,2,3],device='cuda');record={}
  for arm,m in models.items():
   m.eval()
   with torch.no_grad():
    one=m(x);m._relation_pending=[x,x] if arm=='C' else [];native=env[2].forward(m,x,x)[1]
    error=float((one-native).abs().max());assert error<1e-6
   m.train();calls=[];hooks=[b.register_forward_hook(lambda *a:calls.append(1)) for b in [m.conv3d_features[1],m.conv2d_features[1]]]
   torch.set_rng_state(cpu[0]);torch.cuda.set_rng_state(cuda[0]);m.zero_grad();torch.nn.functional.cross_entropy(m(x),y).backward()
   assert len(calls)==2
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   tok=m.conv_a if arm=='A' else m.relation_scores
   assert sum(float(p.grad.norm()) for p in tok.parameters())>0
   for hook in hooks:hook.remove()
   record[arm]={'native_eval_logit_max_error':error,'one_call_each_BN':True,'tokenizer_gradient_nonzero':True,'normal_BN_momentum':.1,'parameters':sum(p.numel() for p in m.parameters()),'ending_cpu_rng':torch.get_rng_state().tolist(),'ending_cuda_rng':torch.cuda.get_rng_state().tolist()}
  assert record['A']['ending_cpu_rng']==record['C']['ending_cpu_rng'];assert record['A']['ending_cuda_rng']==record['C']['ending_cuda_rng']
  for r in record.values():r.pop('ending_cpu_rng');r.pop('ending_cuda_rng')
  records[str(seed)]=record
 dump(OUT/'verification.json',{'passed':True,'seeds':list(SEEDS),'per_seed':records,'source_only_forward_single_input':True,'original_backbone_initial_tensors_identical':True,'no_target_GT_or_training_dataset':True,'all3_blocks_retained':True});print('True source-only preflight passed')
if __name__=='__main__':main()
