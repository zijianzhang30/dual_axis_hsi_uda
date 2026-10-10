import argparse,json,hashlib,time
import torch,numpy as np
import torch.nn.functional as F
import study as s
from readout import install,base_hash

def train(arm,seed):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);torch.cuda.set_device(0);c=s.environment();m,ls,helper,classes,ba=c.setup('houston','C',seed);source,val,target,ev=ls;old=json.loads((s.baseline(seed)/'config.json').read_text());assert ba['complete_initial_hash']==old['initial_full_model_sha256'] and ba['indices_sha256']==old['indices_sha256'];ia=install(m,arm,seed);assert base_hash(m)==old['initial_full_model_sha256']
 target.dataset.labels=np.zeros_like(target.dataset.labels);target.dataset.label=np.zeros_like(target.dataset.label);m.cuda();opt=torch.optim.SGD(m.parameters(),lr=.01);assert {id(p) for p in m.parameters()}=={id(p) for g in opt.param_groups for p in g['params']};assert sum(p.numel() for p in m.parameters())==378347
 folder=s.OUT/('%s_%d'%(arm,seed));folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();config={'plan':s.PLAN,'arm':arm,'seed':seed,'base_audit':ba,'new_module_initial_audit':ia,'initial_complete_hash':c.state_hash(m),'parameters':378347,'baseline_parameters':378011,'added_trainable_parameters':336,'projection_buffer_elements':3072 if arm=='R2' else 0,'projection':'fixed buffer,not optimizer' if arm=='R2' else None};s.dump(folder/'config.json',config)
 ref=[json.loads(x) for x in (s.baseline(seed)/'history.jsonl').read_text().splitlines()];rows=[];cps=[];start=time.monotonic();calls={};hooks=[]
 for name,mod in [('block_%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
  def make(name):
   def h(mod,args,out):calls[name]=calls.get(name,0)+1
   return h
  hooks.append(mod.register_forward_hook(make(name)))
 with (folder/'history.jsonl').open('x') as history:
  for epoch in range(1,201):
   m.train();calls.clear();bh=hashlib.sha256();rh=hashlib.sha256();steps=count=correct=0;loss_sum=0.;first=None
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for a in (x,y,t):bh.update(a.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes());x,y,t=x.cuda(),y.cuda(),t.cuda();opt.zero_grad(set_to_none=True);out=m(x,t);loss=F.cross_entropy(out[0],y);assert torch.isfinite(loss);loss.backward();assert m.center_readout.weight.grad is not None and torch.isfinite(m.center_readout.weight.grad).all()
    if steps==0:first={'new_readout_gradient_norm':float(m.center_readout.weight.grad.norm()),'original_classifier_gradient_norm':float(m.nn1.weight.grad.norm()),'new_weight_norm_before_update':float(m.center_readout.weight.norm()),'normal_main_forward_readout_diagnostics':dict(m._readout_diagnostics)}
    opt.step();n=len(y);steps+=1;count+=n;correct+=int((out[0].argmax(1)==y).sum());loss_sum+=float(loss.detach())*n
   assert steps==18;assert bh.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('pairing_failure_STOP',arm,seed,epoch)
   bn={n:int(v.num_batches_tracked) for n,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==18*epoch for v in bn.values());assert all(calls['block_%d'%i]==18 for i in range(3)) and calls['classifier']==72
   r={'epoch':epoch,'steps':steps,'source_CE':loss_sum/count,'source_train_OA':correct/count*100,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'BN_counts':bn,'main_forward_calls':dict(calls),'first_training_batch_diagnostic':first,'new_readout_weight_norm':float(m.center_readout.weight.norm()),'target_GT_read':False,'elapsed_seconds':time.monotonic()-start}
   if epoch==1 or epoch%10==0:r['source_val']=helper.source_evaluate(m,val,torch.device('cuda:0'));print(json.dumps(r),flush=True)
   if epoch%10==0:
    cp=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'epoch':epoch,'config':config},cp);cps.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp)})
   history.write(json.dumps(r)+'\n');history.flush()
 for h in hooks:h.remove()
 s.verify();s.dump(folder/'frozen_manifest.json',{'arm':arm,'seed':seed,'completed_epochs':200,'steps':3600,'checkpoints':cps,'train_seconds':time.monotonic()-start,'source_val_final':r['source_val'],'no_target_GT_scored':True,'all200_data_RNG_hashes_match_R0':True,'BN_Transformer_calls_unchanged':True});print('CENTER_READOUT_TRAIN_COMPLETE',arm,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);a=p.parse_args();train(a.arm,a.seed)
