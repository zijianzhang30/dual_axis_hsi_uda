import argparse, copy, json, time
import numpy as np
import torch
import study as s
from pairing import Pairing

def preflight():
 torch.set_num_threads(2);c,ns=s.environment();c.seed_worker(2100);args=c.options(2100);args.device='0'
 values,audit=ns['loaders'](args);src,_,tar,ev=values
 tar.dataset.label=np.zeros_like(tar.dataset.label);tar.dataset.labels=np.zeros_like(tar.dataset.labels)
 p=Pairing(2100,len(tar.dataset));p.epoch=2
 ids=np.arange(128);y=np.arange(128)%12
 assert np.array_equal(p.select(y,ids)[0],ids),'empty cache must preserve all rows'
 z=np.full((len(tar.dataset),12),-12.);z[np.arange(len(z)),np.arange(len(z))%12]=12
 p.observe(np.arange(len(z)),z);picked,matched,pred,_=p.select(y,ids)
 assert matched.all() and np.array_equal(pred,y)
 # Remove C2 target candidates: source C2 must remain, falling back rather than being resampled.
 p.valid[np.arange(len(z))%12==1]=False
 picked,matched,pred,_=p.select(y,ids)
 assert np.all(picked[y==1]==ids[y==1]) and not np.any(matched[y==1])
 assert np.array_equal(y,np.arange(128)%12)
 p.epoch=1;assert np.array_equal(p.select(y,ids)[0],ids)
 original=c.helper.state_hash
 model,ema,initial=c.build(args,'A');model=model.cuda();ema=ema.cuda()
 config=json.loads((s.BASE/'seed_2100/config.json').read_text())
 assert initial==config['initial_common_model_sha256'] and original(model)==config['initial_complete_state_sha256']
 assert original(ema)==config['initial_EMA_state_sha256']
 # Independent cache prediction must leave model parameters/buffers/modes and global RNG untouched.
 modes=[m.training for m in model.modules()];gen_state=src.generator.get_state().clone()
 small=copy.copy(ev.dataset);small.indices=small.indices[:128];small.labels=small.labels[:128]
 tiny=Pairing(2100,128);before=original(model);tiny.refresh(model,small,2100)
 assert original(model)==before and [m.training for m in model.modules()]==modes
 assert torch.equal(gen_state,src.generator.get_state())
 # Zero-change controls cover logits, gradients/SGD, native BN counters and teacher paired inputs.
 x=torch.stack([src.dataset[i][0] for i in range(16)]).cuda()
 t=torch.stack([tar.dataset[i][0] for i in range(16)]).cuda();labels=torch.tensor([int(src.dataset[i][1]) for i in range(16)],device='cuda')
 student0=copy.deepcopy(model);student1=copy.deepcopy(model)
 cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();results=[]
 class ZeroTrace:pass
 zero=ZeroTrace();zero.phase='train';zero.source=(x.cpu(),labels.cpu(),torch.arange(16));zero.target=(t.cpu(),torch.arange(16))
 disabled=Pairing(2100,len(tar.dataset),enabled=False);disabled.dataset=tar.dataset;disabled.trace=zero
 handle=student1.register_forward_pre_hook(disabled.pre)
 for net in (student0,student1):
  before_bn=[int(m.num_batches_tracked) for m in net.modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]
  net.train();torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
  outputs=net(x,t.clone());loss=torch.nn.functional.cross_entropy(outputs[0],labels)
  opt=torch.optim.SGD(net.parameters(),lr=.01);loss.backward();opt.step()
  after_bn=[int(m.num_batches_tracked) for m in net.modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]
  assert all(b-a==2 for a,b in zip(before_bn,after_bn))
  results.append((outputs,original(net),torch.cuda.get_rng_state().clone()))
 handle.remove()
 assert all(torch.equal(a,b) for a,b in zip(results[0][0],results[1][0]))
 assert results[0][1]==results[1][1] and torch.equal(results[0][2],results[1][2])
 # Actual pairing hook with nonempty bank, finite backward and same student/EMA target tensor.
 pair=Pairing(2100,len(tar.dataset));pair.dataset=tar.dataset;pair.epoch=2;pair.observe(np.arange(len(z)),z)
 class Dummy:pass
 trace=Dummy();trace.phase='train';trace.source=(x.cpu(),labels.cpu(),torch.arange(16));trace.target=(t.cpu(),torch.arange(16));pair.trace=trace
 incoming=t.clone();before_target=incoming.clone();pair.pre(student1,(x,incoming))
 assert not torch.equal(incoming,before_target)
 with torch.no_grad():
  ema.train();teacher_inputs=[]
  check=ema.register_forward_pre_hook(lambda m,ins:teacher_inputs.append(ins[1].detach().clone()))
  ema(x,incoming);check.remove()
  assert torch.equal(teacher_inputs[0],incoming)
 assert len(pair.rows)==1
 student1.zero_grad();out=student1(x,incoming);torch.nn.functional.cross_entropy(out[0],labels).backward()
 assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().max()>0 for v in student1.parameters())
 # Original release MMD self-self bug is deliberately preserved; no loss AST changes.
 assert 'out_x_fusion_src, out_x_fusion_src' in (c.BIDA/'train_pipeline.py').read_text()
 s.dump(s.OUT/'preflight.json',{'passed':True,'source_batch_not_resampled':True,'empty_class_fallback_checked':True,
 'epoch1_original_pairing':True,'class_matching_checked':True,'cache_no_model_BN_RNG_mutation':True,
 'global_source_loader_rng_unchanged':True,'zero_change_logits_updates_RNG_equal':True,
 'actual_pairing_forward_backward_finite':True,'student_EMA_paired_input_identical':True,'native_BN_updates_two_per_forward':True,'original_MMD_bug_preserved':True,
 'max_zero_change_logit_error':0.0,'initial_student_teacher_hash_match_B0':True,'target_values_zeroed':True})
 print('EQ9_PREFLIGHT_PASS',flush=True)

def train(seed):
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify_originals()
 torch.set_num_threads(2);torch.cuda.set_device(0);c,ns=s.environment();c.seed_worker(seed);args=c.options(seed);args.device='0'
 values,data=ns['loaders'](args);source,val,target,target_eval=values
 old=json.loads((s.BASE/('seed_%d/config.json'%seed)).read_text());assert data==old['data_protocol']
 geom=s.geometry();coords,mapping,missing,extra,full_labels=geom.full_geometry(target_eval.dataset)
 target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels)
 model,ema,initial=c.build(args,'A');assert initial==old['initial_common_model_sha256']
 assert c.helper.state_hash(model)==old['initial_complete_state_sha256'] and c.helper.state_hash(ema)==old['initial_EMA_state_sha256']
 model=model.cuda();ema=ema.cuda();optimizer,scheduler=c.load_scheduler('BiDA',model,args);assert scheduler is None
 criterion,unused=c.make_loss(args,12)
 folder=s.OUT/('B1_%d'%seed);folder.mkdir(exist_ok=False)
 for name in ('checkpoints','predictions','pairing'): (folder/name).mkdir()
 s.dump(folder/'config.json',{'plan':s.PLAN,'settings':vars(args),'data_protocol':data,'seed':seed,
 'initial_student':c.helper.state_hash(model),'initial_teacher':c.helper.state_hash(ema),
 'parameters':sum(p.numel() for p in model.parameters()),'additional_parameters':0})
 reference=[json.loads(x) for x in (s.BASE/('seed_%d/history.jsonl'%seed)).read_text().splitlines()]
 pair=Pairing(seed,len(target.dataset));pair.dataset=target.dataset
 BaseTrace=ns['Trace']
 class Trace(BaseTrace):
  def after_validation(self,epoch,acc,metrics):
   super().after_validation(epoch,acc,metrics)
   before=c.helper.state_hash(self.model);rng=self.rng()
   clean=np.empty((10317,12),np.float32);clean[mapping]=np.concatenate(self.logits);clean[missing]=geom.extra_logits(self.model,extra)
   assert before==c.helper.state_hash(self.model) and rng==self.rng()
   path=folder/'predictions'/('full_epoch_%03d.npz'%epoch);np.savez_compressed(path,logits=clean,predictions=clean.argmax(1))
   record=self.predictions[-1];record['paper_full_metrics']=c.helper.metrics(full_labels,clean.argmax(1),clean,12)
   record['full_prediction_path']=str(path);record['full_prediction_sha256']=s.sha(path)
  def end_epoch(self,epoch,global_step):
   expected=reference[epoch-1]
   assert self.batch.hexdigest()==expected['batch_stream_sha256'],('source/candidate augmentation stream',seed,epoch)
   assert self.old_rng.hexdigest()==expected['rng_stream_sha256'],('CPU/CUDA RNG',seed,epoch)
   assert self.indices.hexdigest()==expected['actual_centers_sha256'],('source/candidate index stream',seed,epoch)
   assert self.rng_digest.hexdigest()==expected['all_rng_stream_sha256'],('all baseline RNG streams',seed,epoch)
   before_rng=self.rng();before_mode=[m.training for m in self.model.modules()]
   pair.refresh(self.model,target_eval.dataset,seed,reuse=np.concatenate(self.logits) if epoch%10==0 else None)
   assert before_rng==self.rng() and before_mode==[m.training for m in self.model.modules()]
   diagnostics=pair.save_epoch(folder/'pairing'/('epoch_%03d.npz'%epoch))
   self.history.write(json.dumps({'pairing_epoch':epoch,'pairing':diagnostics})+'\n')
   assert diagnostics['source_CE_class_counts'][1]>0,'C2 source CE coverage removed'
   if epoch==1:assert diagnostics['matched_rows']==0 and abs(self.ce/self.count-expected['source_ce'])<1e-9
   if epoch in (100,200):
    torch.save({'model':model.state_dict(),'EMA':ema.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
      'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),'loader_rng':source.generator.get_state(),
      'numpy_rng':np.random.get_state(),'python_rng':s.random.getstate(),'pair_cache_probs':pair.probs,'pair_cache_valid':pair.valid,
      'pair_sampling_rng':pair.pick.get_state(),'pair_augmentation_rng':pair.aug.get_state()},folder/'checkpoints'/('resumable_epoch_%03d.pth'%epoch))
   pair.epoch=epoch+1;pair.reset_counts()
   super().end_epoch(epoch,global_step)
 trace=Trace(model,ema,source.generator,torch.device('cuda:0'),folder,source.dataset,target.dataset);pair.trace=trace
 # Pairing modifies the existing target tensor in place: student AND EMA receive identical paired views.
 hooks=[model.register_forward_pre_hook(pair.pre),model.register_forward_pre_hook(trace.pre),
 model.register_forward_hook(trace.post),model.register_forward_hook(pair.post),ema.register_forward_pre_hook(trace.teacher_pre)]
 with (folder/'history.jsonl').open('w') as h,(folder/'actual_batches.jsonl').open('w') as b:
  trace.history=h;trace.batch_log=b;native=ns['native'](trace);start=time.monotonic()
  native(model,ema,optimizer,criterion,12,ns['Loader'](source,trace,'source'),val,ns['Loader'](target,trace,'target'),
   ns['Loader'](target_eval,trace,'eval'),args,str(folder),torch.device('cuda:0'),scheduler,trace)
 for hook in hooks:hook.remove()
 s.dump(folder/'frozen_manifest.json',{'config':json.loads((folder/'config.json').read_text()),'completed_epochs':200,
 'train_seconds':time.monotonic()-start,'evaluations':trace.predictions,'total_steps':16000,
 'all_source_and_original_candidate_order_augmentation_RNG_streams_match_B0':True,
 'actual_selected_target_differs_by_design':True,'source_CE_class_coverage_preserved':True,'only_pairing_changed':True})
 s.verify_originals();print('B1_COMPLETE seed=%d'%seed,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--seed',type=int,choices=s.SEEDS);args=parser.parse_args()
 if args.preflight:preflight()
 else:assert args.seed is not None;train(args.seed)
