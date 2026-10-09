import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
import torch.nn.functional as F
import study as s

def weight(epoch):return .3*(2/(1+math.exp(-10*epoch/200))-1)

def preflight():
 torch.set_num_threads(2);c,ns=s.environment();lm=s.lmmd();rows=[];hashes=[]
 for arm in s.ARMS:
  c.seed_worker(2100);args=c.options(2100);args.device='0';m,unused,initial=s.build(c,args,arm);hashes.append(s.shared_hash(m))
  expected=json.loads((s.BASE/'seed_2100/config.json').read_text());assert initial==expected['initial_common_model_sha256']
  if arm=='C':
   old=json.loads((s.PRIOR/'frozen_ours_2100/config.json').read_text());assert c.helper.state_hash(m)==old['initial_student']
  m=m.cuda().train();capture=[]
  hook=m.norm.register_forward_hook(lambda mod,inp,out:capture.append(out[:,0]))
  x=torch.randn(8,1,176,13,13,device='cuda');t=torch.randn_like(x,requires_grad=True);y=torch.arange(8,device='cuda')%12
  before={k:v.clone() for k,v in m.state_dict().items()};cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
  out=m(x,t);assert len(capture)==4
  ce=F.cross_entropy(out[0],y);losslm=lm.lmmd(capture[0],capture[1],y,out[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=8)
  g=torch.autograd.grad(losslm,list(m.parameters()),retain_graph=True,allow_unused=True);norm=sum(float(v.square().sum()) for v in g if v is not None)**.5;assert norm>0
  total=ce+weight(1)*losslm;assert torch.isfinite(total).all();total.backward()
  assert t.grad is not None and t.grad.abs().sum()>0
  if arm=='C':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.relation_scores.parameters());assert m.relation_scores.relations(x)[...,6,6].abs().max()==0
  # Weight-zero produces exactly the CE path with the same model/input/dropout state.
  m.load_state_dict(before);capture.clear();torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);again=m(x,t)
  assert torch.equal(again[0],out[0]) and torch.equal(F.cross_entropy(again[0],y),ce.detach())
  assert int(m.conv3d_features[1].num_batches_tracked)==int(before['conv3d_features.1.num_batches_tracked'])+1
  hook.remove();c.seed_worker(2100);values,data=ns['loaders'](args)
  ds=copy.copy(values[-1].dataset);ds.indices=ds.indices[:128];ds.labels=ds.labels[:128]
  kernel=s.module('joint_eq9_pair',s.OUT/'pairing_kernel.py');pair=kernel.Pairing(2100,128);pair.refresh(m,ds,2100)
  pools,_,_,_=pair.pools();assert pair.valid.all()
  dummy=kernel.Pairing(2100,128);dummy.epoch=2;source=np.arange(128)%12;ids=np.arange(128);sel,matched,_,_=dummy.select(source,ids)
  assert np.array_equal(sel,ids) and not matched.any()
  rows.append({'arm':arm,'shared_initial_hash':hashes[-1],'parameters':sum(p.numel() for p in m.parameters()),
  'LMMD_gradient_norm':norm,'LMMD_value':float(losslm.detach()),'zero_weight_logit_error':0.0,'BN_updates_per_step':1,
  'target_gradient_present':True,'cache_BN_params_RNG_unchanged':True,'source_coverage_empty_pool_fallback':True})
  del m,unused,capture,out,again,g,total,x,t
 assert hashes[0]==hashes[1]
 s.dump(s.OUT/'preflight.json',{'passed':True,'cases':rows,'schedule_samples':{str(e):weight(e) for e in (1,100,200)},
 'loss_inputs':'ONLY source/target SELF normCLS; no coupled outputs/EMA/MMD', 'pairing_kernel_matches_B1':s.sha(s.OUT/'pairing_kernel.py')==s.sha(s.B1/'experiment_code/pairing.py')})
 print('JOINT_LMMD_EQ9_PREFLIGHT_PASS',flush=True)

def train(seed,arm):
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify();torch.set_num_threads(2);torch.cuda.set_device(0)
 c,ns=s.environment();lm=s.lmmd();args=c.options(seed);args.device='0';c.seed_worker(seed)
 values,data=ns['loaders'](args);source,val,target,ev=values
 assert data==json.loads((s.BASE/('seed_%d/config.json'%seed)).read_text())['data_protocol']
 # Verify actual dataset pool hashes without iterating or creating an extra loader.
 assert {str(i):hashlib.sha256(loader.dataset.indices.tobytes()).hexdigest() for i,loader in enumerate(values)}==data['index_sha256']
 geom=s.geometry();coords,mapping,missing,extra,unused_gt=geom.full_geometry(ev.dataset);del unused_gt
 for ds in (target.dataset,ev.dataset):ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
 m,unused_teacher,initial=s.build(c,args,arm);shared=s.shared_hash(m);m=m.cuda();optimizer=torch.optim.SGD(m.parameters(),lr=.01)
 # Keep parent CE+Eq9 unused native loss initialization so random streams match that parent.
 criterion,unused_center=c.make_loss(args,12)
 folder=s.OUT/('%s_%d'%(arm,seed));folder.mkdir(exist_ok=False)
 for name in ('checkpoints','predictions','pairing'):(folder/name).mkdir()
 s.dump(folder/'config.json',{'plan':s.PLAN,'arm':arm,'seed':seed,'settings':vars(args),'data_protocol':data,
 'initial_student':c.helper.state_hash(m),'shared_initial_hash':shared,'initial_original_hash':initial,'parameters':sum(p.numel() for p in m.parameters())})
 reference=[json.loads(x) for x in (s.PRIOR/('frozen_ours_%d/history.jsonl'%seed)).read_text().splitlines()]
 pair=s.module('joint_eq9_pair',s.OUT/'pairing_kernel.py').Pairing(seed,len(target.dataset));pair.dataset=target.dataset
 class Trace(ns['Trace']):
  def evaluate(self,epoch,loader):
   assert self.steps==80 and self.teacher_calls==0;self.phase='validation';self.logits=[];self.labels=[]
   before=c.helper.state_hash(m);m.eval()
   with torch.inference_mode():
    for x,dummy in loader:
     assert not dummy.any();x=x.cuda();m(x,x)
   native=np.concatenate(self.logits);full=np.empty((10317,12),np.float32);full[mapping]=native;full[missing]=geom.extra_logits(m,extra)
   assert before==c.helper.state_hash(m)
   record={'epoch':epoch}
   for kind,logits in [('native',native),('full',full)]:
    p=folder/'predictions'/('%s_epoch_%03d.npz'%(kind,epoch));np.savez_compressed(p,logits=logits,predictions=logits.argmax(1));record[kind+'_prediction_path']=str(p);record[kind+'_sha256']=s.sha(p)
   p=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'epoch':epoch},p);record['checkpoint_path']=str(p);record['checkpoint_sha256']=s.sha(p)
   self.predictions.append(record);self.phase='train'
  def end_epoch(self,epoch,global_step):
   assert self.steps==80 and global_step==80*epoch and self.teacher_calls==0
   expected=reference[epoch-1];assert self.batch.hexdigest()==expected['batch_stream_sha256'];assert self.indices.hexdigest()==expected['actual_centers_sha256']
   assert self.old_rng.hexdigest()==expected['rng_stream_sha256'];assert self.rng_digest.hexdigest()==expected['all_rng_stream_sha256']
   before_rng=self.rng();pair.refresh(m,ev.dataset,seed,reuse=np.concatenate(self.logits) if epoch%10==0 else None);assert before_rng==self.rng()
   diag=pair.save_epoch(folder/'pairing'/('epoch_%03d.npz'%epoch));assert diag['source_CE_class_counts'][1]>0
   counts={k:int(v.num_batches_tracked) for k,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==epoch*80 for v in counts.values())
   row={'epoch':epoch,'global_step':global_step,'steps':80,'source_ce':self.ce/self.count,'LMMD':self.lm_total/self.count,
    'LMMD_weight':weight(epoch),'source_train_OA':self.hits/self.count*100,'pairing':diag,'BN_batches':counts,
    'batch_stream_sha256':self.batch.hexdigest(),'actual_centers_sha256':self.indices.hexdigest(),
    'rng_stream_sha256':self.old_rng.hexdigest(),'all_rng_stream_sha256':self.rng_digest.hexdigest(),
    'target_GT_scoring':False,'elapsed_seconds':time.monotonic()-start}
   self.history.write(json.dumps(row)+'\n');self.history.flush();self.batch_log.flush()
   if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
   if epoch in (100,200):torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
    'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),'loader_rng':source.generator.get_state(),
    'numpy_rng':np.random.get_state(),'python_rng':s.random.getstate(),'pair_probs':pair.probs,'pair_valid':pair.valid,
    'pair_pick_rng':pair.pick.get_state(),'pair_aug_rng':pair.aug.get_state()},folder/'checkpoints'/('resumable_epoch_%03d.pth'%epoch))
   self.previous_end={'student':ns['state'](m),'teacher':ns['state'](unused_teacher)};self.epoch=epoch+1;self.reset();self.lm_total=0.;self.hits=0
   pair.epoch=epoch+1;pair.reset_counts()
 trace=Trace(m,unused_teacher,source.generator,torch.device('cuda:0'),folder,source.dataset,target.dataset);trace.lm_total=0.;trace.hits=0;pair.trace=trace
 capture=[]
 def capture_cls(module,inputs,output):
  if m.training and trace.phase=='train':capture.append(output[:,0])
 hooks=[m.register_forward_pre_hook(pair.pre),m.register_forward_pre_hook(trace.pre),m.register_forward_hook(trace.post),m.register_forward_hook(pair.post),m.norm.register_forward_hook(capture_cls)]
 with (folder/'history.jsonl').open('w') as h,(folder/'actual_batches.jsonl').open('w') as b:
  trace.history=h;trace.batch_log=b;start=time.monotonic();global_step=0
  sl=ns['Loader'](source,trace,'source');tl=ns['Loader'](target,trace,'target');el=ns['Loader'](ev,trace,'eval')
  for epoch in range(1,201):
   m.train()
   for (x,y),(t,dummy) in zip(sl,tl):
    if len(x)!=len(t):continue
    assert not dummy.any();x,y,t=x.cuda(),y.cuda(),t.cuda();optimizer.zero_grad(set_to_none=True);capture.clear()
    r=m(x,t);assert len(capture)==4
    ce=F.cross_entropy(r[0],y);lmd=lm.lmmd(capture[0],capture[1],y,r[1].softmax(1),CLASS_NUM=12,BATCH_SIZE=128)
    total=ce+weight(epoch)*lmd;assert torch.isfinite(total).all();total.backward();optimizer.step();global_step+=1
    trace.lm_total+=float(lmd.detach())*len(y);trace.hits+=int((r[0].argmax(1)==y).sum())
   if epoch%10==0:trace.evaluate(epoch,el)
   trace.end_epoch(epoch,global_step)
 for hook in hooks:hook.remove()
 s.dump(folder/'frozen_manifest.json',{'arm':arm,'seed':seed,'config':json.loads((folder/'config.json').read_text()),'completed_epochs':200,
 'total_steps':16000,'train_seconds':time.monotonic()-start,'evaluations':trace.predictions,'all_predictions_frozen_before_target_scoring':True})
 print('COMPLETE_UNSCORED',arm,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--seed',type=int,choices=s.SEEDS);p.add_argument('--arm',choices=s.ARMS);a=p.parse_args()
 if a.preflight:preflight()
 else:assert a.seed and a.arm;train(a.seed,a.arm)
