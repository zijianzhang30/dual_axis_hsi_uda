import argparse,hashlib,json,math,time
import numpy as np
import torch
import torch.nn.functional as F
import study as s
import adapter as a

def train(arm,seed):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);torch.cuda.set_device(0)
 c=s.environment();model,loaders,helper,classes,audit=c.setup('hyrank','C',seed);source,val,target,ev=loaders
 old=json.loads((s.HIST/'hyrank'/('C_LMMD_%d/config.json'%seed)).read_text());assert audit['complete_initial_hash']==old['complete_initial_hash'] and audit['indices_sha256']==old['indices_sha256']
 a.install(model,arm,seed);target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels)
 model=model.cuda();native=c.lmmd();capture=[];hook=model.norm.register_forward_hook(lambda mod,inp,out:capture.append(out[:,0]))
 optimizer=torch.optim.SGD(model.parameters(),lr=.01);folder=s.OUT/('%s_%d'%(arm,seed));folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir()
 config={'plan':s.PLAN,'arm':arm,'seed':seed,'base_audit':audit,'parameters':sum(p.numel() for p in model.parameters()),
 'extra_parameters':6144 if arm=='R1' else 12288,'per_path_width':16,'reference':str(s.HIST/'hyrank'/('C_LMMD_%d'%seed))}
 s.dump(folder/'config.json',config)
 reference=[json.loads(x) for x in (s.HIST/'hyrank'/('C_LMMD_%d/history.jsonl'%seed)).read_text().splitlines()]
 start=time.monotonic();checkpoints=[]
 initial={k:p.detach().clone() for k,p in model.named_parameters() if '.adapter_' in k}
 with (folder/'history.jsonl').open('x') as history:
  for epoch in range(1,201):
   model.train();bhash=hashlib.sha256();rhash=hashlib.sha256();steps=count=hits=0;ces=lms=0.;grad_sums={k:0. for k in initial};grad_steps={k:0 for k in initial}
   weight=.3*(2/(1+math.exp(-10*epoch/200))-1)
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bhash.update(v.numpy().tobytes())
    rhash.update(torch.get_rng_state().numpy().tobytes());rhash.update(torch.cuda.get_rng_state().numpy().tobytes())
    x,y,t=x.cuda(),y.cuda(),t.cuda();optimizer.zero_grad(set_to_none=True);capture.clear();out=model(x,t);assert len(capture)==4
    ce=F.cross_entropy(out[0],y);lmd=native.lmmd(capture[0],capture[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=len(y))
    loss=ce+weight*lmd;assert torch.isfinite(loss).all();loss.backward()
    grads=a.gradients(model)
    for k,v in grads.items():grad_sums[k]+=v;grad_steps[k]+=int(v>0)
    optimizer.step();steps+=1;n=len(y);count+=n;hits+=int((out[0].argmax(1)==y).sum());ces+=float(ce.detach())*n;lms+=float(lmd.detach())*n
   assert steps==80
   expected=reference[epoch-1];assert bhash.hexdigest()==expected['batch_stream_sha256'],('data',arm,seed,epoch)
   assert rhash.hexdigest()==expected['model_rng_sha256'],('dropout RNG',arm,seed,epoch)
   BN={k:int(m.num_batches_tracked) for k,m in model.named_modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(n==epoch*80 for n in BN.values())
   row={'epoch':epoch,'steps':80,'source_ce':ces/count,'LMMD':lms/count,'LMMD_weight':weight,'source_train_OA':hits/count*100,
    'batch_stream_sha256':bhash.hexdigest(),'model_rng_sha256':rhash.hexdigest(),'BN_batches':BN,'target_scoring':False,
    'adapter_gradient_mean_norm':{k:v/steps for k,v in grad_sums.items()},'adapter_gradient_nonzero_steps':grad_steps,
    'adapter_parameter_change_norm':{k:float((p.detach()-initial[k]).norm()) for k,p in model.named_parameters() if k in initial}}
   if epoch==1 or epoch%10==0:
    with a.eval_domain(model,'source'):row['source_val']=helper.source_evaluate(model,val,torch.device('cuda:0'))
    row['source_val_adapter_role']='source';row['elapsed_seconds']=time.monotonic()-start
    path=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':s.sha(path)})
    print(json.dumps(row),flush=True)
   history.write(json.dumps(row)+'\n');history.flush()
 hook.remove();s.dump(folder/'frozen_manifest.json',{'arm':arm,'seed':seed,'completed_epochs':200,'total_steps':16000,'config':config,'checkpoints':checkpoints,
 'train_seconds':time.monotonic()-start,'source_val_at_final':row['source_val'],'target_GT_not_scored_during_training':True})
 print('FORMAL_ADAPTER_TRAIN_COMPLETE',arm,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);args=p.parse_args();train(args.arm,args.seed)
