import json,hashlib,importlib.util
from pathlib import Path
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');OUT=B/'houston_center_spectrum_readout_v1';SEEDS=(2100,2101,2102);ARMS=('R1','R2')
PLAN={'dataset':'Houston13->Houston18','seeds':list(SEEDS),'epochs':200,'batch':128,'patch':13,'bands':48,'network':'formal C + Full Joint BN + original3 SelfTransformer/classifier','R0':'reuse matching formal C: WzZ+b','R1':'WzZ+b+Wx xc; xc=actual preprocessed and augmented input[:,0,:,6,6]; Linear48->7 no bias, weight zero','R2':'WzZ+b+Wa PZ; fixed48x64 orthonormal-row projection generated from independent CPU generator(seed+117000); Linear48->7 no bias, weight zero','R2_scope':'extra336parameters/optimization-parameterization control; equivalent linear decision function,not extra expressivity','new_trainable_parameters_each':336,'R2_fixed_projection_buffer_elements':3072,'new_module_init':'torch.random.fork_rng(devices=[]), private RNG seed+116000; assert original CPU and CUDA states unchanged','loss':'source-self CE only','excluded':'LMMD,Eq9,adapter,coupled losses,distillation,EMA,MCC,additionalviews','optimizer':'persistent SGD lr0.01 momentum0 weight_decay0 no schedule; new weight included','data':'unchanged normband per-pixelL2, source95% native2393/127,target52901 native candidate mask; target class values zero before any training batch','training_budget':'18 updates/epoch,3600 total; native source-val epoch1/every10; paired data,augmentation,modeldropoutRNG checked everyepoch vs R0','evaluation_grid':list(range(10,201,10)),'formal':'fixed200; no checkpoint selection','Best_OA':'target-oracle diagnostic,identical20-checkpointgrid; AA/Kappa at corresponding OA-best epoch,do not mix','continue_rule':'R1-R0 andR1-R2 averageOA/AA positive andmajority seeds bothpositive,review severe single-seed orC5/C6/C7/classcoverage degradation; fixed andBest separate,automatic followup prohibited','maximum_new_trainings':6,'Pavia_HyRANK_not_started':True,'target_hyperparameter_tuning':False}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('center_readout_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline(seed):return B/'tokenizer_mechanism_v2/houston'/('C_%d'%seed)
def verify():
 environment().verify_files()
 for name in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p
