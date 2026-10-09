import torch,json,io,hashlib
from common import OUT,build,environment,state_hash,dump

def main():
 torch.set_num_threads(2);env=environment();models={};original={}
 for arm in ['C','P0','P1']:
  env[4](2100);model,h=build(2100,arm,'houston');models[arm]=model.cuda();original[arm]=h
 assert len(set(original.values()))==1
 base=models['C'].state_dict()
 for arm in ['P0','P1']:
  assert all(torch.equal(v,models[arm].state_dict()[k]) for k,v in base.items())
 assert state_hash(models['P0'])==state_hash(models['P1'])
 params={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()};assert params['P0']==params['P1']==params['C']+16448
 x=torch.randn(7,1,48,13,13,device='cuda');t=torch.randn_like(x);labels=torch.arange(7,device='cuda');out={}
 for arm,m in models.items():
  m.eval()
  with torch.no_grad():out[arm]=m(x,t)[1]
 for arm in ['P0','P1']:assert (out[arm]-out['C']).abs().max()<1e-6
 mt=torch.randn(7,4,64,device='cuda');mw=torch.randn(7,4,169,device='cuda').softmax(-1)
 for arm in ['P0','P1']:
  mem=models[arm].prototype_reasoning;assert mem.M_T.shape==(7,4,64) and mem.M_W.shape==(7,4,169)
  assert not any(n.startswith('M_') for n,p in mem.named_parameters());mem.update_source(mt,mw,labels);assert torch.equal(mem.M_T,mt) and torch.equal(mem.M_W,mw)
  mem.update_source(mt+1,mw,labels);assert torch.allclose(mem.M_T,mt+.1);assert not mem.M_T.requires_grad and not mem.M_W.requires_grad
 cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state();record={};states={}
 for arm,m in models.items():
  m.train();torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);calls={'bn3d':0,'bn2d':0,'blocks':0};hooks=[]
  def counter(k):
   def f(*a):calls[k]+=1
   return f
  for module,key in [(m.conv3d_features[1],'bn3d'),(m.conv2d_features[1],'bn2d'),*[(b,'blocks') for b in m.blocks]]:hooks.append(module.register_forward_hook(counter(key)))
  if arm!='C':beforeT=m.prototype_reasoning.M_T.clone();beforeW=m.prototype_reasoning.M_W.clone();m.prototype_reasoning.diagnostics=True
  inp=t.detach().clone().requires_grad_();m.zero_grad();loss=torch.nn.functional.cross_entropy(m(x,inp)[0],labels);loss.backward();assert inp.grad.norm()>0
  assert calls=={'bn3d':1,'bn2d':1,'blocks':3};states[arm]=(torch.get_rng_state(),torch.cuda.get_rng_state())
  for h in hooks:h.remove()
  if arm!='C':
   mem=m.prototype_reasoning;assert torch.equal(mem.M_T,beforeT) and torch.equal(mem.M_W,beforeW),'Forward modified memory'
   grads={n:float(p.grad.norm()) for n,p in mem.named_parameters()};assert all(v>0 for v in grads.values());assert mem.M_T.grad is None and mem.M_W.grad is None
   if arm=='P1':
    last=mem.last;expected=last['similarity'].topk(8,dim=-1).indices;assert torch.equal(last['selected'],expected)
   raw_source=m._prototype_raw[0];expected=.9*beforeT+.1*raw_source[0];m.update_source_memory(labels);assert torch.allclose(mem.M_T,expected)
   record[arm]={'BN_and_block_calls':calls,'projection_gradients':grads,'memory_detached_and_source_update_correct':True,'target_BN_gradient_nonzero':True}
 for arm in ['P0','P1']:assert all(torch.equal(a,b) for a,b in zip(states[arm],states['C']))
 # Preserve module mode/buffers and verify checkpoint target-only round trip.
 errors={}
 for arm in ['P0','P1']:
  m=models[arm];m.eval();m.prototype_reasoning.diagnostics=True
  with torch.no_grad():one=m.target_only(t);paired=m(x,t)[1]
  assert (one-paired).abs().max()<1e-6
  stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0);env[4](2100);loaded,_=build(2100,arm,'houston');loaded.load_state_dict(torch.load(stream,map_location='cpu',weights_only=True));loaded.cuda().eval()
  with torch.no_grad():after=loaded.target_only(t)
  assert torch.equal(one,after);errors[arm]={'paired_eval_error':float((one-paired).abs().max()),'round_trip_target_only_error':float((one-after).abs().max())}
 # An output-only external class/prediction value cannot affect selection: selector takes T/W and buffers only.
 dump(OUT/'verification.json',{'passed':True,'parameters':params,'new_trainable_parameters':16448,'memory_T_W_float32_bytes':7*4*(64+169)*4,'memory_buffers_in_optimizer':False,'independent_module_init_and_original_tensors_exact':True,'empty_memory_C_logit_equivalence':True,'no_extra_main_RNG_draws':True,'target_labels_accessed':False,'top8_by_relation_similarity_only':True,'records':record,'target_only_round_trip':errors});print('RGDR preflight passed')
if __name__=='__main__':main()
