import argparse,copy,hashlib,json,time
import numpy as np
import torch
import torch.nn.functional as F
import study as s

def preflight():
 torch.set_num_threads(2);c,ns=s.environment();args=c.options(2100);args.device='0'
 result=[]
 for track in s.TRACKS:
  c.seed_worker(2100);m,ema,initial=s.build(c,args,track)
  old=json.loads((s.BASE/'seed_2100/config.json').read_text());assert initial==old['initial_common_model_sha256']
  if track=='frozen_ours':
   c.seed_worker(2100);ref,_,_=c.build(args,'C')
   assert all(torch.equal(v,ref.state_dict()[k]) for k,v in m.state_dict().items())
   assert m.conv3d_features[1].momentum==m.conv2d_features[1].momentum==.19
  else:assert m.conv3d_features[1].momentum==m.conv2d_features[1].momentum==.1
  m=m.cuda();ema=ema.cuda();x=torch.randn(4,1,176,13,13,device='cuda');t=torch.randn_like(x,requires_grad=True);y=torch.tensor([0,1,2,3],device='cuda')
  if track=='frozen_ours':
   ref=ref.cuda();m.eval();ref.eval()
   with torch.no_grad():err=float((m(x,t)[1]-ref(x,t)[1]).abs().max())
   assert err==0
  else:err=None
  m.train();before=[int(m.conv3d_features[1].num_batches_tracked),int(m.conv2d_features[1].num_batches_tracked)]
  out=m(x,t);loss=F.cross_entropy(out[0],y)
  if track=='bida_full_C':
   with torch.no_grad():eo=ema(x,t)
   loss+=.1*(c.official.distill_loss(out[2],out[1])+c.official.distill_loss(out[3],out[0]))
   loss+=(c.official.softmax_mse_loss(F.softmax(out[0],dim=1),F.softmax(eo[0],dim=1))+c.official.softmax_mse_loss(F.softmax(out[1],dim=1),F.softmax(eo[1],dim=1)))/4
   assert all(not p.requires_grad for p in ema.parameters())
  loss.backward();assert torch.isfinite(loss)
  assert sum(float(p.grad.abs().sum()) for p in m.relation_scores.parameters() if p.grad is not None)>0
  assert sum(float(p.grad.abs().sum()) for p in m.conv3d_features.parameters() if p.grad is not None)>0
  if track=='frozen_ours':assert t.grad is not None and float(t.grad.abs().sum())>0
  after=[int(m.conv3d_features[1].num_batches_tracked),int(m.conv2d_features[1].num_batches_tracked)]
  assert [b-a for a,b in zip(before,after)]==([1,1] if track=='frozen_ours' else [2,2])
  weights=m.relation_scores(x);assert weights.shape==(4,4,169) and torch.allclose(weights.sum(-1),torch.ones(4,4,device='cuda'))
  rel=m.relation_scores.relations(x);assert rel[...,6,6].abs().max()==0
  # Read-only cache prediction preserves all parameters/buffers/modes and global RNG.
  c.seed_worker(2100);values,data=ns['loaders'](args);ds=copy.copy(values[-1].dataset);ds.indices=ds.indices[:128];ds.labels=ds.labels[:128]
  pair=s.module('ours_eq9_pair',s.OUT/'pairing_kernel.py').Pairing(2100,128)
  pair.refresh(m,ds,2100);assert pair.valid.all()
  result.append({'track':track,'parameters':sum(p.numel() for p in m.parameters()),'backbone_init_matches_B0_B1':True,
   'MLP_independent_RNG':True,'token_shape':[4,4,64],'weight_shape':list(weights.shape),'center_anchor':[6,6],
   'nonzero_relation_and_CNN_gradients':True,'target_gradient_via_joint_BN':track=='frozen_ours',
   'BN_updates_per_student_forward':[b-a for a,b in zip(before,after)],'cache_eval_changes_no_BN_or_RNG':True,
   'frozen_ours_vs_historical_C_eval_logit_error':err,'teacher_parameters_detached':track=='bida_full_C'})
  del m,ema,x,t,out
 s.dump(s.OUT/'preflight.json',{'passed':True,'cases':result,'same_B1_pairing_kernel_sha256':s.sha(s.OUT/'pairing_kernel.py')})
 print('TWO_TRACK_PREFLIGHT_PASS',flush=True)

def train(seed,track):
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify();torch.set_num_threads(2);torch.cuda.set_device(0)
 c,ns=s.environment();args=c.options(seed);args.device='0';c.seed_worker(seed)
 values,data=ns['loaders'](args);source,val,target,ev=values
 old=json.loads((s.BASE/('seed_%d/config.json'%seed)).read_text());assert data==old['data_protocol']
 geom=s.geometry();coords,mapping,missing,extra,full_labels=geom.full_geometry(ev.dataset)
 target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels)
 model,ema,initial=s.build(c,args,track);assert initial==old['initial_common_model_sha256']
 model=model.cuda()
 if track=='bida_full_C':ema=ema.cuda()
 optimizer,scheduler=c.load_scheduler('BiDA',model,args);assert scheduler is None
 # Preserve native unused CenterLoss construction/CPU RNG even on the source-CE-only track.
 criterion,unused=c.make_loss(args,12)
 folder=s.OUT/('%s_%d'%(track,seed));folder.mkdir(exist_ok=False)
 for name in ('checkpoints','predictions','pairing'):(folder/name).mkdir()
 s.dump(folder/'config.json',{'plan':s.PLAN,'track':track,'seed':seed,'settings':vars(args),'data_protocol':data,
 'initial_original_backbone_sha256':initial,'initial_student':c.helper.state_hash(model),'initial_teacher':c.helper.state_hash(ema),
 'parameters':sum(p.numel() for p in model.parameters()),'BN_momentum':.19 if track=='frozen_ours' else .1})
 references=[json.loads(x) for x in (s.BASE/('seed_%d/history.jsonl'%seed)).read_text().splitlines()]
 pair=s.module('ours_eq9_pair',s.OUT/'pairing_kernel.py').Pairing(seed,len(target.dataset));pair.dataset=target.dataset
 class Trace(ns['Trace']):
  def before_validation(self,epoch):
   assert self.steps==80 and self.teacher_calls==(80 if track=='bida_full_C' else 0)
   self.phase='validation';self.logits=[];self.labels=[]
   self.eval_before={'student':ns['state'](self.model),'teacher':ns['state'](self.ema),'rng':self.rng()}
  def after_validation(self,epoch,acc,metrics):
   super().after_validation(epoch,acc,metrics)
   before=c.helper.state_hash(self.model);rng=self.rng()
   full=np.empty((10317,12),np.float32);full[mapping]=np.concatenate(self.logits);full[missing]=geom.extra_logits(model,extra)
   assert before==c.helper.state_hash(self.model) and rng==self.rng()
   path=folder/'predictions'/('full_epoch_%03d.npz'%epoch);np.savez_compressed(path,logits=full,predictions=full.argmax(1))
   record=self.predictions[-1];record['paper_full_metrics']=c.helper.metrics(full_labels,full.argmax(1),full,12)
   record['full_prediction_path']=str(path);record['full_prediction_sha256']=s.sha(path)
  def end_epoch(self,epoch,global_step):
   assert self.steps==80 and global_step==epoch*80
   expected=references[epoch-1]
   assert self.batch.hexdigest()==expected['batch_stream_sha256'],('source/candidate augmentation',track,seed,epoch)
   assert self.indices.hexdigest()==expected['actual_centers_sha256'],('source/candidate indices',track,seed,epoch)
   if track=='bida_full_C':
    assert self.old_rng.hexdigest()==expected['rng_stream_sha256'],('CUDA/CPU RNG',track,seed,epoch)
    assert self.rng_digest.hexdigest()==expected['all_rng_stream_sha256'],('all RNG',track,seed,epoch)
   # Frozen CE-only intentionally has no EMA dropout draws. Same CPU/loader/Python/NumPy stream still required.
   for key in ('shared_loader','torch_cpu','python','numpy'):
    assert self.last_forward[key]==expected['last_training_forward_rng'][key],(key,track,seed,epoch)
   before_rng=self.rng();pair.refresh(model,ev.dataset,seed,reuse=np.concatenate(self.logits) if epoch%10==0 else None)
   assert before_rng==self.rng()
   diag=pair.save_epoch(folder/'pairing'/('epoch_%03d.npz'%epoch));assert diag['source_CE_class_counts'][1]>0
   row={'epoch':epoch,'global_step':global_step,'steps':self.steps,'source_ce':self.ce/self.count,'pairing':diag,
    'batch_stream_sha256':self.batch.hexdigest(),'actual_centers_sha256':self.indices.hexdigest(),
    'rng_stream_sha256':self.old_rng.hexdigest(),'all_rng_stream_sha256':self.rng_digest.hexdigest(),
    'source_candidate_stream_matches_B0_B1':True,'CPU_loader_python_numpy_RNG_matches_B0_B1':True,
    'CUDA_RNG_matches_B1':track=='bida_full_C','teacher_calls':self.teacher_calls}
   if epoch%10==0:row['target']=self.predictions[-1]['metrics'];row['full_target']=self.predictions[-1]['paper_full_metrics']
   if epoch in (100,200):
    torch.save({'model':model.state_dict(),'EMA':ema.state_dict() if track=='bida_full_C' else None,'optimizer':optimizer.state_dict(),
    'epoch':epoch,'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),'loader_rng':source.generator.get_state(),
    'numpy_rng':np.random.get_state(),'python_rng':s.random.getstate(),'pair_probs':pair.probs,'pair_valid':pair.valid,
    'pair_rng':pair.pick.get_state(),'pair_aug_rng':pair.aug.get_state()},folder/'checkpoints'/('resumable_epoch_%03d.pth'%epoch))
   self.history.write(json.dumps(row)+'\n');self.history.flush();self.batch_log.flush()
   if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
   self.previous_end={'student':ns['state'](model),'teacher':ns['state'](ema)}
   self.epoch=epoch+1;self.reset();pair.epoch=epoch+1;pair.reset_counts()
 trace=Trace(model,ema,source.generator,torch.device('cuda:0'),folder,source.dataset,target.dataset);pair.trace=trace
 hooks=[model.register_forward_pre_hook(pair.pre),model.register_forward_pre_hook(trace.pre),model.register_forward_hook(trace.post),model.register_forward_hook(pair.post)]
 if track=='bida_full_C':hooks.append(ema.register_forward_pre_hook(trace.teacher_pre))
 with (folder/'history.jsonl').open('w') as h,(folder/'actual_batches.jsonl').open('w') as b:
  trace.history=h;trace.batch_log=b;start=time.monotonic()
  sl=ns['Loader'](source,trace,'source');tl=ns['Loader'](target,trace,'target');el=ns['Loader'](ev,trace,'eval')
  if track=='bida_full_C':
   native=ns['native'](trace);native(model,ema,optimizer,criterion,12,sl,val,tl,el,args,str(folder),torch.device('cuda:0'),scheduler,trace)
  else:
   global_step=0
   for epoch in range(1,201):
    model.train()
    for (x,y),(t,_) in zip(sl,tl):
     if len(x)!=len(t):continue
     x,y,t=x.cuda(),y.cuda(),t.cuda();optimizer.zero_grad()
     loss=F.cross_entropy(model(x,t)[0],y);assert torch.isfinite(loss)
     loss.backward();optimizer.step();global_step+=1
    if epoch%10==0:
     trace.before_validation(epoch);acc,metrics=c.official.validation(model,el,torch.device('cuda:0'),12,show=True);trace.after_validation(epoch,acc,metrics)
    trace.end_epoch(epoch,global_step)
 for hook in hooks:hook.remove()
 s.dump(folder/'frozen_manifest.json',{'track':track,'seed':seed,'config':json.loads((folder/'config.json').read_text()),
 'completed_epochs':200,'train_seconds':time.monotonic()-start,'total_steps':16000,'evaluations':trace.predictions})
 print('TRACK_COMPLETE',track,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--seed',type=int,choices=s.SEEDS);p.add_argument('--track',choices=s.TRACKS);args=p.parse_args()
 if args.preflight:preflight()
 else:assert args.seed is not None and args.track;train(args.seed,args.track)
