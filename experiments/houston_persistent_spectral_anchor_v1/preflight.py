import copy,json,torch
import torch.nn.functional as F
import study as s
from anchor import install,base_hash

def main():
 s.verify();torch.set_num_threads(2);torch.cuda.set_device(0);c=s.environment();m,ls,helper,n,ba=c.setup('houston','C',s.SEED)
 install(m,'S0');a=copy.deepcopy(m);audit=install(a,'S2');m.cuda();a.cuda()
 # Fixed synthetic paired patches and source labels; no target GT read.
 gen=torch.Generator().manual_seed(812600);x=torch.rand(8,1,48,13,13,generator=gen).cuda();t=torch.rand(8,1,48,13,13,generator=gen).cuda();y=(torch.arange(8)%7).cuda()
 initcpu=torch.get_rng_state().clone();initgpu=torch.cuda.get_rng_state().clone();opts=[torch.optim.SGD(v.parameters(),lr=.01) for v in (m,a)];results=[]
 for model,opt in zip((m,a),opts):
  torch.set_rng_state(initcpu);torch.cuda.set_rng_state(initgpu);model.train();opt.zero_grad(set_to_none=True);out=model(x,t);F.cross_entropy(out[0],y).backward()
  results.append({'logits':[v.detach().clone() for v in out],'grads':{n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters() if not n.startswith('spectral_')},'cpu':torch.get_rng_state().clone(),'gpu':torch.cuda.get_rng_state().clone(),'bn':{n:v.clone() for n,v in model.state_dict().items() if 'running_' in n or 'num_batches_tracked' in n}})
  opt.step()
 p,q=results;assert all(torch.equal(x,y) for x,y in zip(p['logits'],q['logits']))
 assert p['grads'].keys()==q['grads'].keys()
 assert all((p['grads'][n] is None and q['grads'][n] is None) or (p['grads'][n] is not None and q['grads'][n] is not None and torch.equal(p['grads'][n],q['grads'][n])) for n in p['grads']),[(n,float((p['grads'][n]-q['grads'][n]).abs().max())) for n in p['grads'] if p['grads'][n] is not None and not torch.equal(p['grads'][n],q['grads'][n])]
 assert torch.equal(p['cpu'],q['cpu']) and torch.equal(p['gpu'],q['gpu']);assert all(torch.equal(p['bn'][n],q['bn'][n]) for n in p['bn']);assert base_hash(m)==base_hash(a)
 alpha_grad=a.spectral_alpha.grad.detach().cpu().tolist();assert all(abs(v)>0 for v in alpha_grad);alpha_step=a.spectral_alpha.detach().cpu().tolist()
 opts[1].zero_grad(set_to_none=True);F.cross_entropy(a(x,t)[0],y).backward();eg=float(a.spectral_encoder[0].weight.grad.norm());assert eg>0
 # Zero-alpha eval equivalence also after matched base SGD update.
 with torch.no_grad():
  a.spectral_alpha.zero_()
  # Second disposable gradient probe updated its BN buffers; restore matching
  # original tensors before checking zero-alpha eval equivalence.
  for k,v in m.state_dict().items():a.state_dict()[k].copy_(v)
 m.eval();a.eval()
 with torch.no_grad():eo=m(x,t)[1];ea=a(x,t)[1]
 assert torch.equal(eo,ea)
 # Ordinary token shape and same encoder init checked on fresh disposable CPU models.
 b,_,_,_,_=c.setup('houston','C',s.SEED);d=copy.deepcopy(b);install(b,'S1');install(d,'S2');assert all(torch.equal(v,d.spectral_encoder.state_dict()[k]) for k,v in b.spectral_encoder.state_dict().items())
 s.dump(s.OUT/'preflight.json',{'passed':True,'train_all_four_logits_bitwise_equal':True,'eval_logits_bitwise_equal':True,'all_original_base_gradients_bitwise_equal':True,'original_base_SGD_update_bitwise_equal':True,'CPU_CUDA_RNG_bitwise_equal':True,'BN_running_stats_and_update_counts_bitwise_equal':True,'alpha_zero_first_gradient':alpha_grad,'alpha_after_first_SGD':alpha_step,'spectral_encoder_nonzero_gradient_second_step':eg,'S1_S2_identical_encoder_initialization':True,'preflight_disposable_models_only':True,'target_GT_read':False,'new_module':audit})
 print('PREFLIGHT_PASS',json.loads((s.OUT/'preflight.json').read_text()),flush=True)
if __name__=='__main__':main()
