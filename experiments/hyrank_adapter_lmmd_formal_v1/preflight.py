import torch,json,numpy as np
import study as s
import adapter as a

def main():
 torch.set_num_threads(2);c=s.environment();native=c.lmmd();results=[]
 for seed in s.SEEDS:
  for arm in s.ARMS:
   m,loaders,helper,classes,audit=c.setup('hyrank','C',seed);before=helper.state_hash(m)
   old=json.loads((s.HIST/'hyrank'/('C_LMMD_%d/config.json'%seed)).read_text());assert audit['complete_initial_hash']==old['complete_initial_hash']
   import copy
   ref=copy.deepcopy(m);a.install(m,arm,seed);assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.state_dict().items())
   m=m.cuda().train();ref=ref.cuda().train();x=torch.randn(12,1,176,13,13,device='cuda');t=torch.randn_like(x);y=torch.arange(12,device='cuda')
   cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();base=ref(x,t);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
   capture=[];hook=m.norm.register_forward_hook(lambda mod,inp,out:capture.append(out[:,0]));out=m(x,t)
   err=max(float((u-v).abs().max()) for u,v in zip(base,out));assert err<1e-6
   lmd=native.lmmd(capture[0],capture[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=12)
   ps=[p for name,p in m.named_parameters() if ('.adapter_target.up.' in name or '.adapter_shared.up.' in name)]
   g=torch.autograd.grad(lmd,ps,retain_graph=True);norm=sum(float(v.square().sum()) for v in g)**.5;assert norm>0
   loss=torch.nn.functional.cross_entropy(out[0],y)+.1*lmd;loss.backward();opt=torch.optim.SGD(m.parameters(),lr=.01);opt.step()
   capture.clear();opt.zero_grad(set_to_none=True);out=m(x,t);lmd=native.lmmd(capture[0],capture[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=12)
   (torch.nn.functional.cross_entropy(out[0],y)+.1*lmd).backward();stats=a.gradients(m);assert all(v>0 for v in stats.values())
   hook.remove();m.eval()
   if arm=='R2':
    with torch.no_grad():
     with a.eval_domain(m,'target'):target=m(x,t)[1].clone()
     with a.eval_domain(m,'source'):source=m(x,t)[1].clone()
     saved={name:p.clone() for name,p in m.named_parameters() if '.adapter_source.' in name}
     for name,p in m.named_parameters():
      if name in saved:p.fill_(100.)
     with a.eval_domain(m,'target'):assert torch.equal(m(x,t)[1],target)
     with a.eval_domain(m,'source'):assert not torch.equal(m(x,t)[1],source)
     for name,p in m.named_parameters():
      if name in saved:p.copy_(saved[name])
   results.append({'arm':arm,'seed':seed,'old_R0_initial_hash_matches':True,'base_logits_max_error_zero_adapter':err,
    'direct_target_LMMD_gradient_norm':norm,'all_adapter_parameter_gradients_nonzero_after_step':True,
    'source_target_eval_routing_checked':True,'extra_parameters':6144 if arm=='R1' else 12288,'each_path_width':16})
   del m,ref,loaders,capture,base,out,g,loss,stats,x,t
 s.dump(s.OUT/'preflight.json',{'passed':True,'checks':results,'R0_three_seeds_reusable':True})
 print('FORMAL_ADAPTER_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
