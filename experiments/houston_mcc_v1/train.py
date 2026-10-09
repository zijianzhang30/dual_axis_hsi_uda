import argparse,json,hashlib,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
from mcc_loss import MinimumClassConfusionLoss
from diagnostics import diagnostic

def train(seed):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);torch.cuda.set_device(0);c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',seed);source,val,target,ev=ls
 old=json.loads((s.baseline(seed)/'config.json').read_text());assert audit['complete_initial_hash']==old['initial_full_model_sha256'] and audit['indices_sha256']==old['indices_sha256']
 target.dataset.labels=np.zeros_like(target.dataset.labels);target.dataset.label=np.zeros_like(target.dataset.label);m=m.cuda();mcc=MinimumClassConfusionLoss(temperature=2.);opt=torch.optim.SGD(m.parameters(),lr=.01)
 assert sum(p.numel() for p in m.parameters())==old['parameters'];assert {id(p) for p in m.parameters()}=={id(p) for g in opt.param_groups for p in g['params']}
 folder=s.OUT/('MCC_%d'%seed);folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();s.dump(folder/'config.json',{'plan':s.PLAN,'seed':seed,'base_audit':audit,'parameters':old['parameters'],'CE_reference':str(s.baseline(seed)),'official_snapshot':str(s.OUT/'official_snapshot/upstream.json')})
 ref=[json.loads(x) for x in (s.baseline(seed)/'history.jsonl').read_text().splitlines()];checkpoints=[];start=time.monotonic();calls={};hooks=[]
 for name,mod in [('Transformer_block_%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1),('CLS_norm',m.norm)]:
  def make(name):
   def h(mod,inp,out):calls[name]=calls.get(name,0)+1
   return h
  hooks.append(mod.register_forward_hook(make(name)))
 with (folder/'history.jsonl').open('x') as h:
  for epoch in range(1,201):
   m.train();calls.clear();bhash=hashlib.sha256();rhash=hashlib.sha256();steps=count=hits=0;ce_sum=mcc_sum=entropy_sum=temp_entropy_sum=0.;mat=np.zeros((7,7));raw=np.zeros((7,7));gradient=None
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bhash.update(v.numpy().tobytes())
    rhash.update(torch.get_rng_state().numpy().tobytes());rhash.update(torch.cuda.get_rng_state().numpy().tobytes());x,y,t=x.cuda(),y.cuda(),t.cuda();opt.zero_grad(set_to_none=True);out=m(x,t);ce=F.cross_entropy(out[0],y);mc=mcc(out[1].float());loss=ce+mc;assert mc.dtype==torch.float32 and torch.isfinite(loss).all(),'NaN/Inf STOP, no automatic loss repair'
    if steps==0:
     ps=list(m.parameters());gm=torch.autograd.grad(mc,ps,retain_graph=True,allow_unused=True);gc=torch.autograd.grad(ce,ps,retain_graph=True,allow_unused=True);gt=torch.autograd.grad(mc,out[1],retain_graph=True)[0];nm=sum(float(v.square().sum()) for v in gm if v is not None)**.5;nc=sum(float(v.square().sum()) for v in gc if v is not None)**.5
     assert torch.isfinite(gt).all() and nm>0;gradient={'sample':'actual first training batch of this epoch,same main forward','MCC_norm':nm,'CE_norm':nc,'weighted_MCC_over_CE':nm/nc,'target_logits_MCC_norm':float(gt.norm())};del gm,gc,gt
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step();n=len(y);steps+=1;count+=n;hits+=int((out[0].argmax(1)==y).sum());ce_sum+=float(ce.detach())*n;mcc_sum+=float(mc.detach())*n
    with torch.no_grad():d=diagnostic(out[1]);entropy_sum+=d['target_entropy']*n;temp_entropy_sum+=d['MCC_temperature_entropy']*n;mat+=np.asarray(d['normalized_confusion']);raw+=np.asarray(d['raw_confusion'])
   assert steps==18 and bhash.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rhash.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('stream mismatch',seed,epoch)
   bn={k:int(v.num_batches_tracked) for k,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==epoch*18 for v in bn.values());assert all(calls['Transformer_block_%d'%i]==18 for i in range(3)) and calls['classifier']==calls['CLS_norm']==72
   row={'epoch':epoch,'steps':steps,'source_CE':ce_sum/count,'MCC':mcc_sum/count,'total_loss':(ce_sum+mcc_sum)/count,'target_entropy':entropy_sum/count,'temperature_entropy':temp_entropy_sum/count,'normalized_class_confusion_mean':(mat/steps).tolist(),'raw_class_confusion_mean':(raw/steps).tolist(),'gradient_diagnostic':gradient,'source_train_OA':hits/count*100,'batch_stream_sha256':bhash.hexdigest(),'original_rng_stream_sha256':rhash.hexdigest(),'BN_counts':bn,'main_forward_calls':dict(calls),'target_GT_scored':False}
   if epoch==1 or epoch%10==0:
    row['source_val']=helper.source_evaluate(m,val,torch.device('cuda:0'));row['elapsed_seconds']=time.monotonic()-start
    path=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'epoch':epoch,'config':json.loads((folder/'config.json').read_text())},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':s.sha(path)});print(json.dumps(row),flush=True)
   h.write(json.dumps(row)+'\n');h.flush()
 for hook in hooks:hook.remove()
 s.dump(folder/'frozen_manifest.json',{'seed':seed,'completed_epochs':200,'total_steps':3600,'checkpoints':checkpoints,'train_seconds':time.monotonic()-start,'source_val_final':row['source_val'],'no_target_GT_scored':True});print('MCC_TRAIN_COMPLETE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);train(p.parse_args().seed)
