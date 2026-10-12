import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');CODE=ROOT/'experiments/three_module_factorial_ablation_v1';OUT=ROOT/'results/three_module_factorial_ablation_v1';NAS=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');SEEDS=(2100,2101,2102);DATASETS=('houston','pavia','hyrank');RELATIONS=('N_A','N_C','J_A','J_C')
def module(n,p):
 q=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(q);q.loader.exec_module(v);return v
cross=module('factorial_cross_protocol',ROOT/'experiments/cross_benchmark_relation_identity_dual_evidence_v1/study.py')
houston=module('factorial_houston_protocol',ROOT/'experiments/houston_relation_identity_dual_evidence_3seed_v1/study.py')
sha=cross.sha;dump=cross.dump
PLAN={'datasets':list(DATASETS),'seeds':list(SEEDS),'epochs':200,'grid':list(range(10,201,10)),'factors':['Joint BN off/on','C tokenizer off/on','fixed independent spectral Product-of-Evidence off/on'],'groups':[r+'_'+str(e) for r in RELATIONS for e in (0,1)],'interpretation':'full 2x2x2 factorial, progressive additions and full-model leave-one-out; baseline is self-path source-CE model, NOT full BiDA','BN_off':'native sequential shared BN, native momentum .1, two source/target updates per paired train forward','BN_on':'unchanged Full Joint BN, equal source/target mixture, momentum .19, one joint update per paired forward; momentum is part of established component, disclosed rather than independently estimated','tokenizer_off':'unchanged native learned Conv2d(64,4,1) attention scorer','tokenizer_on':'unchanged original C center-relative normalized nonlinear scorer, fixed hidden32,4tokens','spectral':'reuse immutable independently source-CE trained LN+MLP(bands,64,classes), all bands, same source streams/RNG; no new spectral training required','fusion':'fixed normalized product, float64 logsoftmax sum,T1,no weights/gates/search','loss':'source-self CE ONLY,SGD .01 momentum0 decay0,no scheduler; original three-layer self Transformer; coupled products unused; unused teacher/loss initialization retained only where formal protocol requires','protocols':{'houston':'current original C 18steps/epoch,source-val epoch1/every10','pavia':'repaired-L2 formal C 290steps/epoch,source-val epoch1/every10','hyrank':'current CE-only Eq9-off 80steps/epoch,no source-val; native10287 primary,full10317 secondary'},'streams':'every epoch source/target augmented tensor and original CPU/CUDA RNG digest matches current C; HyRANK also every-batch actual indices and six RNG hashes','reuse':'J_C all9 immutable relation runs; J_A Houston/Pavia6 only after exact200epoch stream/RNG audit; reuse all9 spectral experts/predictions','new_relation_trainings':21,'derived_factorial_endpoints':72,'formal':'fixed200 mean+sample std, same-seed deltaOA/AA/Kappa; per-class recall/zero-recall and recovery/damage, expert disagreement; factorial main effects/interactions descriptive; no significance overclaim with n3','GT':'target label values zero before training/inference; all planned21 runs and predictions globally frozen/hash verified before new-study scoring; historical GT outcomes known','best':'same-grid target-best separately marked oracle only, never checkpoint/parameter/group selection','no_followups':'only locked21 training tasks and necessary evaluation/reporting; no retries/sensitivity/new baselines/new gate automatically','resource':'bounded scheduler GPU4 up to2/GPU3 up to1 jobs; memory guard20GB available; fixed queue, failures recorded and no auto retry'}
PLAN['resource']='bounded scheduler pool GPUs0/2/3/4,max3total/max2perGPU,memory guard20GB,prefer lower current utilization; fixed queue,no auto retry'
def reference(d,seed):return houston.baseline(seed) if d=='houston' else cross.baseline(d,seed)
def reused_relation(d,r,seed):
 if r=='J_C':return reference(d,seed)
 if r=='J_A' and d in ('houston','pavia'):return reference(d,seed).parent/('A_%d'%seed)
 return None
def new_tasks():return [(d,r,k) for d in DATASETS for r in RELATIONS for k in SEEDS if reused_relation(d,r,k) is None]
def setup(d,r,seed):
 import torch,numpy as np
 from argparse import Namespace
 from types import MethodType
 joint=r.startswith('J');ct=r.endswith('C')
 if d in ('houston','pavia'):
  if d=='houston':c,seed_worker,helper=houston.environment().environment('houston');bands=48
  else:c=cross.pavia_env();env=c.environment();seed_worker=env[4];helper=env[6];bands=102
  env=c.environment();seed_worker(seed);args=Namespace(dataset=d,seed=seed,num_workers=4,dataset_dir=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets')/('Houston' if d=='houston' else 'Pavia'))
  if d=='houston':args.dataset_dir=str(args.dataset_dir)+'/'
  values=c.loaders(args);m,initial=c.build(seed,'A',d);old=json.loads((reference(d,seed)/'config.json').read_text());assert initial==old['initial_original_model_sha256'];idx={k:hashlib.sha256(l.dataset.indices.tobytes()).hexdigest() for k,l in zip(('source','val','target','target_eval'),values)};assert idx==old['indices_sha256']
  if d=='pavia':
   for name,ds in zip(('source','target'),(values[0].dataset,values[2].dataset)):assert hashlib.sha256(np.ascontiguousarray(ds.data[6:-6,6:-6]).tobytes()).hexdigest()==old['normalized_cube_sha256'][name]
  if not joint:
   m.forward=MethodType(m._original_forward_func,m);m._tokenize=MethodType(m._original_tokenize_func,m)
   for x in (m.conv3d_features[1],m.conv2d_features[1]):x.momentum=.1
  if ct:__import__('scorer').install(m,'C',seed,bands)
  rt={'hash':c.state_hash,'bands':bands,'classes':7,'steps':18 if d=='houston' else 290,'initial_original':initial,'indices':idx}
 else:
  parent=cross.hyrank_env();c,ns=parent.environment();args=c.options(seed);args.device='0';c.seed_worker(seed);values,data=ns['loaders'](args);old=json.loads((reference(d,seed)/'config.json').read_text());assert data==old['data_protocol'];m,teacher,initial=parent.build(c,args,'A')
  if not joint:
   m.forward=MethodType(m._original_forward_func,m);m._tokenize=MethodType(m._original_tokenize_func,m)
   for x in (m.conv3d_features[1],m.conv2d_features[1]):x.momentum=.1
  if ct:module('factorial_hyrank_C',NAS/'hyrank_joint_lmmd_eq9_v1/relation_176.py').install(m,'C',seed)
  m.cuda();unused_loss=c.make_loss(args,12);rt={'hash':c.helper.state_hash,'bands':176,'classes':12,'steps':80,'data_protocol':data,'teacher':teacher,'unused_loss':unused_loss,'initial_original':initial}
 if r=='J_C':assert rt['hash'](m)==old['initial_full_model_sha256'] if d!='hyrank' else rt['hash'](m)==old['initial_student']
 return m,values,rt
def spec_file(d,seed,epoch,scope='native'):
 if d=='houston':
  f=houston.OUT/('seed_%d'%seed);mf=json.loads((f/'ALL_GRID_PREDICTIONS_FROZEN.json').read_text());rec=next(r for r in mf['records'] if r['epoch']==epoch)
 else:
  f=cross.OUT/('%s_%d'%(d,seed));mf=json.loads((f/'ALL_PREDICTIONS_FROZEN.json').read_text());rec=next(r for r in mf['records'] if r['epoch']==epoch and r['scope']==scope)
 assert sha(rec['path'])==rec['sha256'];return rec
def verify():
 for name in ('code_manifest.json','reuse_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p
