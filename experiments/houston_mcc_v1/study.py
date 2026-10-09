import ast,json,hashlib,importlib.util,shutil
from pathlib import Path
import torch
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');OUT=B/'houston_mcc_v1';SEEDS=(2100,2101,2102)
PLAN={'dataset':'Houston13 -> Houston18','seeds':list(SEEDS),'epochs':200,'temperature':2.,'lambda_mcc':1.,'start_epoch':1,'warmup':False,'model':'formal C+FullJointBN+original3 self Transformer/classifier,NO new layers','objective':'source CE +1.0 official MCC(existing target self original logits out[1])','excluded':'LMMD,Eq9,adapter,EMA,distillation,pseudo CE,multiwindow','protocol':'exact existing C normband48,13patch,bs128,source95%,persistent SGD lr.01/momentum0/decay0,18updates/epoch,sourceval1/every10','target_mask_permission':'existing target nonzero annotation mask for candidate pixels inherited; class values zero before training; no expanded permissions','randomness':'same C initial hash/data/aug/global RNG/model dropout stream;loss no RNG;diagnostic autograd queries no extra forward or optimizer steps','loss_precision':'FP32,official entropy uncertainty weight detached only; probability/logits remain differentiable','official_normalization':'preserve exact class_confusion_matrix / sum(dim=1) broadcasting,do not change axes or add eps beyond official entropy epsilon','evaluation':'fixed200 FORMAL;target-best grid10:10:200 DIAGNOSTIC target-label selection;AA/Kappa same OA epoch','Go':'fixed200 meanOA+AA positive,>=2/3 seeds both positive,no severe degradation or class coverage worsening;target-best only signal insufficient','new_training_limit':3,'no_sweep_or_auto_followups':True,'no_Pavia_HyRANK':True}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('MCC_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py')
def baseline(seed):return B/'tokenizer_mechanism_v2/houston'/('C_%d'%seed)
def official_loss():
 # Compile official nodes directly, independently of transplanted class; no classifier/bottleneck imports.
 ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional}
 for name,cls in [('entropy.py','entropy'),('mcc.py','MinimumClassConfusionLoss')]:
  p=OUT/'official_snapshot'/name;node=next(n for n in ast.parse(p.read_text()).body if getattr(n,'name',None)==cls);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 return ns['MinimumClassConfusionLoss'](temperature=2.)
def verify():
 environment().verify_files()
 for filename in ('code_manifest.json','baseline_manifest.json'):
  for p,h in json.loads((OUT/filename).read_text()).items():assert sha(p)==h,p

def prepare():
 OUT.mkdir(exist_ok=False);code=OUT/'experiment_code';code.mkdir();shutil.copytree(Path(__file__).parent/'official_snapshot',OUT/'official_snapshot')
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 upstream=json.loads((OUT/'official_snapshot/upstream.json').read_text());dump(OUT/'frozen_plan.json',{**PLAN,'official':upstream});history={}
 for seed in SEEDS:
  f=baseline(seed);h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];m=json.loads((f/'frozen_manifest.json').read_text());assert len(h)==200 and all(x['steps']==18 for x in h)
  for p in (f/'config.json',f/'history.jsonl',f/'frozen_manifest.json'):history[str(p)]=sha(p)
  for r in m['checkpoints']+[m['prediction']]:assert sha(r['path'])==r['sha256'];history[r['path']]=r['sha256']
  f=B/'houston_shared_adapter_2x2_v1'/('CE0_%d'%seed);q=f/'offline_predictions_frozen.json';history[str(q)]=sha(q)
  for r in json.loads(q.read_text())['records']:history[r['path']]=r['sha256']
 for p in list((B/'tokenizer_mechanism_v2').glob('*.py'))+list((B/'tokenizer_mechanism_v2/snapshot').rglob('*.py')):history[str(p)]=sha(p)
 dump(OUT/'baseline_manifest.json',history);dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+list((OUT/'official_snapshot').iterdir())});dump(OUT/'history_audit.json',{'no_exact_existing_Houston_MCC_study_found':True,'reuse_C_CE_three_seeds':True,'new_training_count':3})
 (ROOT/'results'/OUT.name).symlink_to(OUT,target_is_directory=True)
if __name__=='__main__':prepare()
