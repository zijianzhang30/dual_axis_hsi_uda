import ast,json,hashlib,importlib.util,shutil,types
from pathlib import Path
import numpy as np,torch
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'houston_cdan_position_v1';SEED=2100;EPOCHS=100
LOCAL=Path(__file__).parent
PLAN={'dataset':'Houston13->Houston18','seed':2100,'epochs':100,'exploratory_fixed_endpoint':100,'groups':['CE historical','LMMD historical','CDAN_Z new','CDAN_T new'],'new_training_count':2,'model':'formal C+FullJointBN+original3 self Transformer/classifier unchanged','protocol':'formal C L2/normband48,patch13,batch128,95%source,18steps/epoch,persistent SGD .01/momentum0/decay0,sourceval1/every10','discriminator':'448->128 ReLU->128 ReLU->1 Sigmoid,no BN/dropout/bottleneck','discriminator_optimizer':'persistent SGD .01,momentum0,decay0;one update per main step','GRL':'official WarmStart formula alpha1,lo0,hi1,max_iters1000;manual step=epoch*18+batch from0;single loss call and critic step regardless K','loss':'CE+domain BCE;feature gradient reversed by GRL only;classifier prediction softmax detached BOTH domains;no entropy weights','T':'actual4x64tokens beforeCLS/position/dropout/Transformer;flatten B*K;shared critic','Z':'actual normalized CLS64 beforeclassifier,self source/target notcoupled','controls':'same critic init and paramcount,base init/data/augmentation/mainforward RNG paired;critic no dropout or BN;no extra mainforward','exclude':'LMMD alongsideCDAN,Eq9,EMA,adapter,distillation,MCC,pseudoCE,newbackbone','report':'fixed100primary;targetbestgrid10:10:100 diagnostic,AA/Kappa atsame OA checkpoint;no target scoring duringtraining','mask_permission':'existing target native nonzero mask only;classvalues zero beforeiteration','no_auto_followups':True,'Pavia_HyRANK_not_started':True}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def environment():
 spec=importlib.util.spec_from_file_location('cdan_frozen_protocol',B/'cross_benchmark_lmmd_v1/snapshot/common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

def baseline(group):return B/'tokenizer_mechanism_v2/houston/C_2100' if group=='CE' else B/'cross_benchmark_lmmd_v1/houston/C_LMMD_2100'
def verify():
 environment().verify_files()
 for p,h in json.loads((OUT/'code_and_baseline_hashes.json').read_text()).items():assert sha(p)==h,p

def official_loss(d):
 # Exact official AST nodes; old np.float alias shim affects dtype alias only,not schedule.
 from typing import Optional,Any,Tuple
 ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'Function':torch.autograd.Function,'Optional':Optional,'Any':Any,'Tuple':Tuple,'np':types.SimpleNamespace(exp=np.exp,float=float),'binary_accuracy':lambda a,b:torch.tensor(0.)}
 for name,nodes in [('entropy.py',['entropy']),('grl.py',['GradientReverseFunction','WarmStartGradientReverseLayer']),('cdan.py',['MultiLinearMap','ConditionalDomainAdversarialLoss'])]:
  tree=ast.parse((OUT/'official_snapshot'/name).read_text())
  for key in nodes:
   node=next(n for n in tree.body if getattr(n,'name',None)==key);exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),ns)
 return ns['ConditionalDomainAdversarialLoss'](d,entropy_conditioning=False,randomized=False)

def prepare():
 OUT.mkdir(exist_ok=False);shutil.copytree(LOCAL/'official_snapshot',OUT/'official_snapshot');shutil.copytree(LOCAL,OUT/'code_snapshot',ignore=shutil.ignore_patterns('__pycache__','official_snapshot','*.log'));dump(OUT/'frozen_plan.json',PLAN);files={}
 for p in (OUT/'code_snapshot').glob('*.py'):files[str(p)]=sha(p)
 for p in (OUT/'official_snapshot').iterdir():files[str(p)]=sha(p)
 for group in ('CE','LMMD'):
  folder=baseline(group)
  for p in (folder/'config.json',folder/'history.jsonl',folder/'frozen_manifest.json'):files[str(p)]=sha(p)
  for e in range(10,101,10):
   p=folder/'checkpoints'/('epoch_%03d.pth'%e);assert p.exists();files[str(p)]=sha(p)
 dump(OUT/'code_and_baseline_hashes.json',files);dump(OUT/'history_audit.json',{'no_matching_completed_C_token_CDan_Z_T_found':True,'CE_LMMD_baselines_reuse_seed2100_epoch100_and_samegrid':True,'baseline_protocol':'same fullinit/cohorts/RNG formal C lineage,noEq9','new_trainings':2})
 link=Path('/home/zhangzj26/dual_axis_hsi_uda/results/houston_cdan_position_v1');link.symlink_to(OUT)
