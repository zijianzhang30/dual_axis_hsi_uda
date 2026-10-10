import argparse,json,hashlib,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
from anchor import install,base_hash

def train(arm):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);torch.cuda.set_device(0);c=s.environment();m,ls,helper,n,ba=c.setup('houston','C',s.SEED);source,val,target,ev=ls;old=json.loads((s.baseline()/'config.json').read_text());ia=install(m,arm);assert base_hash(m)==old['initial_full_model_sha256'];target.dataset.labels=np.zeros_like(target.dataset.labels);target.dataset.label=np.zeros_like(target.dataset.label);m.cuda();opt=torch.optim.SGD(m.parameters(),lr=.01)
 folder=s.OUT/(arm+'_2100');folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();config={'plan':s.PLAN,'arm':arm,'seed':2100,'base_audit':ba,'installation':ia,'parameters':sum(p.numel() for p in m.parameters()),'initial_complete_hash':c.state_hash(m)};s.dump(folder/'config.json',config);ref=[json.loads(x) for x in (s.baseline()/'history.jsonl').read_text().splitlines()]
 calls={};hooks=[]
 for name,mod in [('block_%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
  def make(name):
   def h(mod,args,out):calls[name]=calls.get(name,0)+1
   return h
  hooks.append(mod.register_forward_hook(make(name)))
 cps=[];total=0;start=time.monotonic()
 with (folder/'history.jsonl').open('x') as history:
  for epoch in range(1,101):
   m.train();calls.clear();bh=hashlib.sha256();rh=hashlib.sha256();steps=count=correct=0;loss_sum=0.;first=None
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for a in (x,y,t):bh.update(a.numpy().tobytes())
    cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();rh.update(cpu.numpy().tobytes());rh.update(gpu.numpy().tobytes());x,y,t=x.cuda(),y.cuda(),t.cuda();opt.zero_grad(set_to_none=True)
    out=m(x,t)
    loss=F.cross_entropy(out[0],y);assert torch.isfinite(loss);loss.backward()
    if steps==0:
     first={'base_classifier_grad_norm':float(m.nn1.weight.grad.norm())}
     if arm!='S0':first['encoder_weight_grad_norm']=float(m.spectral_encoder[0].weight.grad.norm())
     if arm=='S3':first['alpha_grad']=m.spectral_alpha.grad.detach().cpu().tolist();first['alpha_before']=m.spectral_alpha.detach().cpu().tolist()
    opt.step();steps+=1;total+=1;count+=len(y);correct+=int((out[0].argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
   assert steps==18 and bh.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('pairing_failure_STOP',arm,epoch,bh.hexdigest(),rh.hexdigest())
   bn={n:int(v.num_batches_tracked) for n,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==18*epoch for v in bn.values());assert all(calls['block_%d'%i]==18 for i in range(3)) and calls['classifier']==72
   row={'epoch':epoch,'steps':18,'source_CE':loss_sum/count,'source_train_OA':correct/count*100,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'BN_counts':bn,'main_forward_calls':dict(calls),'first_batch_gradients':first,'elapsed_seconds':time.monotonic()-start,'target_GT_scored':False}
   if arm=='S3':row['alpha_after_epoch']=m.spectral_alpha.detach().cpu().tolist()
   if epoch==1 or epoch%10==0:row['source_val']=helper.source_evaluate(m,val,torch.device('cuda:0'));print(json.dumps(row),flush=True)
   if epoch%10==0:
    cp=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'epoch':epoch,'config':config},cp);cps.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp)})
   history.write(json.dumps(row)+'\n');history.flush()
 for h in hooks:h.remove()
 extra={}
 if arm=='S3':assert any(r['first_batch_gradients']['encoder_weight_grad_norm']>0 for r in [json.loads(z) for z in (folder/'history.jsonl').read_text().splitlines()][1:])
 s.verify();s.dump(folder/'frozen_manifest.json',{'arm':arm,'seed':2100,'completed_epochs':100,'steps':1800,'checkpoints':cps,'train_seconds':time.monotonic()-start,'data_RNG_all_epochs_match_original_C':True,'BN_Transformer_classifier_call_counts_unchanged':True,'no_target_GT_scored':True,**extra});print('TRAIN_COMPLETE',arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);train(p.parse_args().arm)
