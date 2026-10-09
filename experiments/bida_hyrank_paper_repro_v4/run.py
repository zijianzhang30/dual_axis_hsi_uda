"""Source-release repair only. Existing baselines reused; fixed factor and seeds."""
import argparse
import ast
import copy
import json
import time
import numpy as np
import torch
import study as s

def probe():
 torch.set_num_threads(2);c,ns=s.environment();c.seed_worker(2100);args=c.options(2100);args.device='0';args.num_workers=0
 values,audit=ns['loaders'](args);model,ema,initial=c.build(args,'A')
 assert not hasattr(model,'relation_scores') and not hasattr(model,'_bn_arm')
 assert all(m.momentum==.1 for m in model.modules() if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)))
 batches=[]
 for loader in (values[0],values[2]):
  patches=[];labels=[]
  for i in range(128):
   x,y,ix=loader.dataset[i];patches.append(x);labels.append(y)
  batches.append((torch.stack(patches),torch.tensor(labels)))
 batches[1]=(batches[1][0],torch.zeros_like(batches[1][1]))
 # Pure algebra/gradient test on the actual published MMD implementation.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(610993);a=torch.randn(128,12,requires_grad=True);b=torch.randn(128,12,requires_grad=True)
  mm=c.official.MMD_loss();zero=mm(a,a);gz=torch.autograd.grad(zero,a,retain_graph=True)[0]
  real=mm(a,b);gg=torch.autograd.grad(real,(a,b));assert float(zero)==0 and float(gz.abs().max())==0
  assert float(real)>0 and all(float(g.abs().max())>0 for g in gg)
 def compile_step(source):
  fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train')
  outer=next(n for n in fn.body if isinstance(n,ast.For));inner=next(n for n in outer.body if isinstance(n,ast.For))
  node=ast.parse('def step(network,network_ema,optimizer,criterion,MMD_criterion,data_src,data_tar,opts,device,e,global_step):\n pass').body[0]
  node.body=copy.deepcopy(inner.body)+ast.parse('return loss.detach(), Align_loss.detach() if e>100 else loss.new_zeros(())').body
  ast.fix_missing_locations(node);g=dict(c.official.__dict__);g.update(losses=[],Align_losses=[],Distill_losses=[])
  exec(compile(ast.Module(body=[node],type_ignores=[]),'single_factor_update_probe','exec'),g);return g['step']
 old=compile_step((s.OUT/'released_train_pipeline.py').read_text());new=compile_step((s.OUT/'runtime_snapshot/train_pipeline.py').read_text())
 cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();rows=[]
 for epoch in (1,100,101):
  results=[]
  for step in (old,new):
   student=copy.deepcopy(model).cuda();teacher=copy.deepcopy(ema).cuda();student.train();teacher.train()
   opt=torch.optim.SGD(student.parameters(),lr=.01);torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
   loss,alignment=step(student,teacher,opt,torch.nn.functional.cross_entropy,c.official.MMD_loss(),*batches,args,torch.device('cuda:0'),epoch,epoch*80)
   results.append({'loss':float(loss),'alignment':float(alignment),'student':{k:v.detach().cpu().clone() for k,v in student.state_dict().items()},
       'teacher':{k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()},'rng':torch.cuda.get_rng_state().clone()})
   del student,teacher,opt
  diff=max(float((results[0]['student'][k]-results[1]['student'][k]).abs().max()) for k in results[0]['student'])
  tdiff=max(float((results[0]['teacher'][k]-results[1]['teacher'][k]).abs().max()) for k in results[0]['teacher'])
  assert torch.equal(results[0]['rng'],results[1]['rng'])
  if epoch<=100:assert diff==tdiff==0 and results[0]['loss']==results[1]['loss']
  else:assert diff>0
  rows.append({'schedule_epoch':epoch,'old_loss':results[0]['loss'],'new_loss':results[1]['loss'],'student_max_error':diff,
         'teacher_max_error':tdiff,'CUDA_RNG_identical':True})
 s.dump(s.OUT/'preflight.json',{'passed':True,'zero_self_MMD':float(zero),'zero_self_MMD_gradient_max':float(gz.abs().max()),
      'nonself_MMD':float(real),'cases':rows,'limitation':'initialized models with epoch1/100/101 loss scheduling, not historical epoch100 weights',
      'no_full_baseline_retraining':True,'Ours_models_used':False})
 print('SINGLE_FACTOR_PREFLIGHT_PASS',flush=True)

def train(seed):
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify_originals()
 torch.set_num_threads(2);torch.cuda.set_device(0);c,ns=s.environment();args=c.options(seed);args.device='0';c.seed_worker(seed)
 values,data=ns['loaders'](args);source,val,target,target_eval=values
 old_config=json.loads((s.BASE/('seed_%d/config.json'%seed)).read_text())
 assert data==old_config['data_protocol']
 coords,mapping,missing,extra,full_labels=s.full_geometry(target_eval.dataset)
 target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels)
 model,ema,initial=c.build(args,'A');assert initial==old_config['initial_common_model_sha256']
 assert c.helper.state_hash(model)==old_config['initial_complete_state_sha256']
 assert c.helper.state_hash(ema)==old_config['initial_EMA_state_sha256']
 assert not hasattr(model,'relation_scores') and not hasattr(model,'_bn_arm')
 model=model.cuda();ema=ema.cuda();optimizer,scheduler=c.load_scheduler('BiDA',model,args);assert scheduler is None
 criterion,unused=c.make_loss(args,12)
 folder=s.OUT/('repair_%d'%seed);folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();(folder/'predictions').mkdir()
 s.dump(folder/'config.json',{'plan':s.PLAN,'settings':vars(args),'data_protocol':data,'seed':seed,'initial_student':c.helper.state_hash(model),
          'initial_teacher':c.helper.state_hash(ema),'parameters':sum(p.numel() for p in model.parameters())})
 reference=[json.loads(x) for x in (s.BASE/('seed_%d/history.jsonl'%seed)).read_text().splitlines()]
 baseline_manifest=json.loads((s.BASE/('seed_%d/frozen_manifest.json'%seed)).read_text())
 old_by_epoch={r['epoch']:r for r in baseline_manifest['evaluations']}
 BaseTrace=ns['Trace']
 class Trace(BaseTrace):
  def after_validation(self,epoch,acc,metrics):
   super().after_validation(epoch,acc,metrics)
   before=c.helper.state_hash(self.model);before_rng=self.rng()
   clean=np.empty((10317,12),np.float32);clean[mapping]=np.concatenate(self.logits);clean[missing]=s.extra_logits(self.model,extra)
   assert before==c.helper.state_hash(self.model) and before_rng==self.rng()
   path=folder/'predictions'/('full_epoch_%03d.npz'%epoch);np.savez_compressed(path,logits=clean,predictions=clean.argmax(1))
   record=self.predictions[-1];record['paper_full_metrics']=c.helper.metrics(full_labels,clean.argmax(1),clean,12)
   record['full_prediction_path']=str(path);record['full_prediction_sha256']=s.sha(path)
   if epoch<=100:
    checkpoint=torch.load(old_by_epoch[epoch]['checkpoint_path'],map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v.detach().cpu(),checkpoint[k]) for k,v in self.model.state_dict().items()),('pre-MMD checkpoint diverged',epoch)
    assert ns['state'](self.ema)==old_by_epoch[epoch]['validation_after']['teacher'],('pre-MMD EMA diverged',epoch)
    assert abs(record['metrics']['oa_percent']-old_by_epoch[epoch]['metrics']['oa_percent'])<1e-10
   if epoch in (100,200):
    # Old releases saved only student states. Preserve full new state for any later explicitly requested control.
    torch.save({'model':self.model.state_dict(),'EMA':self.ema.state_dict(),'optimizer':optimizer.state_dict(),
       'epoch':epoch,'CPU_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),
       'loader_rng':source.generator.get_state(),'numpy_rng':np.random.get_state(),
       'python_rng':s.random.getstate(),'student_train':self.model.training,'teacher_train':self.ema.training},
       folder/'checkpoints'/('resumable_epoch_%03d.pth'%epoch))
  def end_epoch(self,epoch,global_step):
   expected=reference[epoch-1]
   assert self.batch.hexdigest()==expected['batch_stream_sha256'],('batch order/augmentation',seed,epoch)
   assert self.old_rng.hexdigest()==expected['rng_stream_sha256'],('CPU/CUDA random stream',seed,epoch)
   assert self.indices.hexdigest()==expected['actual_centers_sha256'],('pixel cohort',seed,epoch)
   assert self.rng_digest.hexdigest()==expected['all_rng_stream_sha256'],('loader/Python/NumPy/CUDA RNG',seed,epoch)
   super().end_epoch(epoch,global_step)
 trace=Trace(model,ema,source.generator,torch.device('cuda:0'),folder,source.dataset,target.dataset)
 hooks=[model.register_forward_pre_hook(trace.pre),model.register_forward_hook(trace.post),ema.register_forward_pre_hook(trace.teacher_pre)]
 with (folder/'history.jsonl').open('w') as h,(folder/'actual_batches.jsonl').open('w') as b:
  trace.history=h;trace.batch_log=b;native=ns['native'](trace);start=time.monotonic()
  native(model,ema,optimizer,criterion,12,ns['Loader'](source,trace,'source'),val,ns['Loader'](target,trace,'target'),ns['Loader'](target_eval,trace,'eval'),args,str(folder),torch.device('cuda:0'),scheduler,trace)
 for hook in hooks:hook.remove()
 s.dump(folder/'frozen_manifest.json',{'config':json.loads((folder/'config.json').read_text()),'completed_epochs':200,'train_seconds':time.monotonic()-start,
         'evaluations':trace.predictions,'total_steps':16000,'full_scene_n':10317,'native_n':10287,'only_one_training_factor_changed':True,
         'pre101_student_and_teacher_exact_replay':True,'all200_data_and_RNG_streams_identical_to_baseline':True})
 s.verify_originals();print('MMD_REPAIR_COMPLETE seed=%d'%seed,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=s.SEEDS);p.add_argument('--probe',action='store_true');p.add_argument('--baseline-eval',action='store_true');a=p.parse_args()
 if a.probe:probe()
 elif a.baseline_eval:assert a.seed;s.baseline_full_eval(a.seed)
 else:assert a.seed;train(a.seed)
