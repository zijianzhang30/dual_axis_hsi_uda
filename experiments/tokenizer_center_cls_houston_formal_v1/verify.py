import torch,json,hashlib
from common import build,environment,OUT,dump,state_hash

def main(seed):
 torch.set_num_threads(2);env=environment();device=torch.device('cuda:0');models={};initial={}
 for arm in ['C','CQ0','CQ1']:
  env[4](seed);m,original=build(seed,arm,'houston');m.to(device);models[arm]=m;initial[arm]={'original':original,'full':state_hash(m),'parameters':sum(p.numel() for p in m.parameters())}
 assert initial['CQ0']==initial['CQ1']
 assert initial['CQ0']['parameters']==initial['C']['parameters']+4096
 base=models['C'].state_dict()
 for arm in ['CQ0','CQ1']:
  s=models[arm].state_dict();assert set(s)-set(base)=={'center_projection.weight'}
  assert all(torch.equal(v,s[k]) for k,v in base.items())
 x=torch.randn(3,1,48,13,13,device=device);t=torch.randn_like(x);y=torch.tensor([0,1,2],device=device)
 result={};eval_outputs={}
 for arm,m in models.items():
  m.eval()
  with torch.no_grad():eval_outputs[arm]=m(x,t)[1].clone()
 for arm in ['CQ0','CQ1']:assert (eval_outputs[arm]-eval_outputs['C']).abs().max()<1e-6
 random=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();outputs={};grads={};ending={};counts={}
 for arm,m in models.items():
  torch.set_rng_state(random);torch.cuda.set_rng_state(cuda);m.train();n={'bn3':0,'bn2':0,'blocks':0};hooks=[]
  def hook(key):
   def f(*args):n[key]+=1
   return f
  for module,key in [(m.conv3d_features[1],'bn3'),(m.conv2d_features[1],'bn2'),*[(b,'blocks') for b in m.blocks]]:hooks.append(module.register_forward_hook(hook(key)))
  m.zero_grad();out=m(x,t)[0];outputs[arm]=out.detach();torch.nn.functional.cross_entropy(out,y).backward()
  grads[arm]={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None};ending[arm]=(torch.get_rng_state(),torch.cuda.get_rng_state());counts[arm]=n
  for h in hooks:h.remove()
  assert n=={'bn3':1,'bn2':1,'blocks':3}
 for arm in ['CQ0','CQ1']:
  assert (outputs[arm]-outputs['C']).abs().max()<1e-6
  assert all(torch.equal(a,b) for a,b in zip(ending[arm],ending['C']))
  assert all((v-grads[arm][k]).abs().max()<1e-6 for k,v in grads['C'].items())
 assert grads['CQ0']['center_projection.weight'].abs().max()==0
 assert grads['CQ1']['center_projection.weight'].norm()>0
 # Verify nonzero projection can condition both actual domain centers and target Joint BN gradient path.
 m=models['CQ1'];m.center_projection.weight.data.copy_(torch.eye(64,device=device)*.01)
 m.zero_grad();target=t.detach().clone().requires_grad_();loss=m(x,target)[0].square().mean();loss.backward();assert target.grad.norm()>0
 assert m.center_projection.weight.grad.norm()>0
 # Deterministic optimizer replay against native C over three training steps.
 for arm in ['C','CQ0']:
  env[4](seed);models[arm],_=build(seed,arm,'houston');models[arm].to(device).train()
 opts={a:torch.optim.SGD(models[a].parameters(),lr=.01) for a in ['C','CQ0']}
 for step in range(3):
  cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
  for arm in ['C','CQ0']:
   torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);opts[arm].zero_grad();torch.nn.functional.cross_entropy(models[arm](x,t)[0],y).backward();opts[arm].step()
  assert all(torch.equal(v,models['CQ0'].state_dict()[k]) for k,v in models['C'].state_dict().items())
 dump(OUT/'verification.json',{'passed':True,'initial':initial,'initial_eval_max_error':{a:float((eval_outputs[a]-eval_outputs['C']).abs().max()) for a in ['CQ0','CQ1']},'counts':counts,'CQ0_three_step_optimizer_replay_bitwise_exact':True,'CQ0_projection_gradient_zero_expected':True,'CQ1_projection_gradient_nonzero':True,'shared_parameter_initial_gradients_equal':True,'CPU_CUDA_rng_equal':True,'target_Joint_BN_gradient_nonzero':True,'actual_value_anchor':[6,6],'parameter_overhead':4096,'target_GT_accessed':False})
 print('All center CLS preflight passed')
if __name__=='__main__':
 records={}
 for seed in [2100,2101,2102]:
  main(seed)
  records[str(seed)]=json.loads((OUT/'verification.json').read_text())
 dump(OUT/'verification.json',{'passed':all(v['passed'] for v in records.values()),'seeds':[2100,2101,2102],'per_seed':records})
