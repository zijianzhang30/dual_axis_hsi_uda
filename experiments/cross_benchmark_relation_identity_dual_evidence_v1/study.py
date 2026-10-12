import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=ROOT/'results/cross_benchmark_relation_identity_dual_evidence_v1';CODE=ROOT/'experiments/cross_benchmark_relation_identity_dual_evidence_v1';SEEDS=(2100,2101,2102);DATASETS=('pavia','hyrank')
PLAN={'datasets':list(DATASETS),'seeds':list(SEEDS),'epochs':200,'grid':list(range(10,201,10)),'R0':'reuse fully trained, immutable exact current C checkpoints and frozen predictions; no new BN/parameter updates','R3':'same Houston Product-of-Evidence: normalize(logsoftmax(rel)+logsoftmax(spec)),float64 stable,T1','spectral':'LayerNorm(all bands)+Linear(bands,64)+GELU+Linear(64,num_classes);same independent CPU fork_rng seed+146000;no dropout or BN','objective':'independent source GT CE,SGD lr.01 momentum0 decay0 no scheduler; no input/head/backbone gradient sharing','pavia_protocol':'formal repaired-L2 tokenizer_mechanism_pavia_l2_fixed_v1 C;102bands7classes,source37236/val1960/target39348,patch13,batch128,4workers,290 steps/epoch,sourceval epoch1/every10','hyrank_protocol':'current unified CE-only Eq9-off: hyrank_ce_eq9_off_2100_v1 C_2100,protocol_finalization_five_runs_v1/CE C_2101/2102;176bands12classes,80steps/epoch,no sourceval calls;native10287 formal;full10317 secondary if available','data_replay':'exact native source/target loader order,augmentation,index/cube hashes; compare every200 epoch source/target tensor fingerprints and original CPU/CUDA RNG digests. Reuse original R0 BN/checkpoints rather than recomputing CNN updates.','RNG_replay':'derive exact original dropout-call shapes/order from disposable real paired C forward, validate RNG equality to dropout-only replay. Advance original CUDA stream by same dropout draws; spectral training no RNG. Native loader/validation iteration cadence preserved.','HyRANK_shadow':'Eq9 disabled: caches never choose inputs. Original eval-only shadow refresh proven RNG/BN neutral; omitted for spectral-only training. No target pseudo-label or probabilities used in training. Keep native checkpoint-eval loader iterations every10. Retain unused original teacher/loss initialization CPU RNG draws.','freeze':'all6 spectral trainings200 and all20grid predictions,coords,agreement hashes frozen before this study reads target labels for scoring; historical C results known','formal':'fixed200 mean+sample std ddof1; target-best same20grid oracle only,no selection for main conclusion','Go':'one benchmark mean deltaOA>0 AND mean deltaAA>0; other benchmark no substantial systematic regression','systematic_regression':'predefined: mean deltaOA<-0.5pp with at least2/3seed deltaOA<0 OR mean deltaAA<-0.5pp with at least2/3seed deltaAA<0; new globalcollapse95/newzero recalls recorded perseed/class','majority':'majority OA/AA positive directions reported, desirable but not extra hard gate beyond user criterion','no_followups':'no new variants,temperature/weights/gates/MoE/hidden tuning; crossbenchmark failure=>Houston-specific,stop; success=>future formal naming/ablation/paper integration only when authorized'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def module(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def baseline(dataset,seed):
 if dataset=='pavia':return B/'tokenizer_mechanism_pavia_l2_fixed_v1/pavia'/('C_%d'%seed)
 return B/'hyrank_ce_eq9_off_2100_v1/C_2100' if seed==2100 else B/'protocol_finalization_five_runs_v1/CE'/('C_%d'%seed)
def pavia_env():
 path=B/'tokenizer_mechanism_pavia_l2_fixed_v1';sys.path.insert(0,str(path));return module('cross_identity_pavia_frozen',path/'common.py')
def hyrank_env():return module('cross_identity_hyrank_current',B/'hyrank_ce_eq9_off_2100_v1/experiment_code/study.py')
def setup(dataset,seed):
 import numpy as np,torch
 from argparse import Namespace
 old=json.loads((baseline(dataset,seed)/'config.json').read_text())
 if dataset=='pavia':
  c=pavia_env();env=c.environment();env[4](seed);args=Namespace(dataset='pavia',seed=seed,num_workers=4,dataset_dir=Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia'));values=c.loaders(args);m,initial=c.build(seed,'C','pavia');helper=env[6]
  assert c.state_hash(m)==old['initial_full_model_sha256'] and initial==old['initial_original_model_sha256'];idx={k:hashlib.sha256(l.dataset.indices.tobytes()).hexdigest() for k,l in zip(('source','val','target','target_eval'),values)};assert idx==old['indices_sha256']
  for domain,ds in zip(('source','target'),(values[0].dataset,values[2].dataset)):
   cube=ds.data[6:-6,6:-6];assert hashlib.sha256(np.ascontiguousarray(cube).tobytes()).hexdigest()==old['normalized_cube_sha256'][domain]
  runtime={'hash':c.state_hash,'helper':helper,'bands':102,'classes':7,'steps':290,'initial':initial,'indices':idx,'dtype':'pavia'}
 else:
  base=hyrank_env();c,ns=base.environment();args=c.options(seed);args.device='0';c.seed_worker(seed);values,data=ns['loaders'](args);assert data==old['data_protocol'];m,teacher,initial=base.build(c,args,'C');assert c.helper.state_hash(m)==old['initial_student'];m.cuda();criterion,unused_center=c.make_loss(args,12)
  runtime={'hash':c.helper.state_hash,'helper':c.helper,'bands':176,'classes':12,'steps':80,'initial':initial,'indices':data['index_sha256'],'ns':ns,'teacher':teacher,'unused_loss':(criterion,unused_center),'data_protocol':data}
 return m,values,runtime

def baseline_records(dataset,seed):
 mf=json.loads((baseline(dataset,seed)/'frozen_manifest.json').read_text())
 if dataset=='pavia':return [{'epoch':r['epoch'],'checkpoint':r['path'],'checkpoint_sha256':r['sha256']} for r in mf['checkpoints']]
 return [{'epoch':r['epoch'],'checkpoint':r['checkpoint_path'],'checkpoint_sha256':r['checkpoint_sha256'],'prediction':r['native_prediction_path'],'prediction_sha256':r['native_sha256'],'full_prediction':r['full_prediction_path'],'full_prediction_sha256':r['full_sha256']} for r in mf['evaluations']]
def verify():
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p
