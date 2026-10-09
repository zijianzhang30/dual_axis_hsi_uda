import ast,hashlib,importlib.util,json,random,shutil,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_joint_lmmd_eq9_v1')
PRIOR=OUT.parent/'hyrank_ours_eq9_two_tracks_v1'
B1=OUT.parent/'bida_hyrank_eq9_pairing_v1'
BASE=OUT.parent/'hyrank_A_no_extra_val_v1'
HIST=OUT.parent/'cross_benchmark_lmmd_v1'
SEEDS=(2100,2101,2102);ARMS=('A','C')
PLAN={'arms':list(ARMS),'seeds':list(SEEDS),'epochs':200,'max_new_training_runs':6,'reuse_complete_matching_runs':0,
 'data':'Frozen HyRANK Dioni->Loukia author out68,176bands,native loader normalization unchanged,source95% hardcoded sampling23,13patch,128batch,80 updates/epoch',
 'optimizer':'persistent SGD lr.01 momentum0 weight_decay0 no scheduler',
 'models':'both Full Joint BN .19 + existing 3-layer self-path Transformer; A original tokenizer; C normalized center-relative MLP bands176 hidden32 K4; no new structure',
 'execution_provenance':'Retain historical BiDA-self implementation and its unused forward products to preserve existing network and dropout RNG. Coupled outputs NEVER enter loss; no coupled training objective, teacher forward/update, distillation, EMA or ordinary MMD.',
 'objective':'source CE + LMMD ONLY',
 'LMMD_schedule':'0.3*(2/(1+exp(-10*epoch/200))-1), starting epoch1; identical prior HyRANK frozen schedule',
 'LMMD_features':'first two actual model.norm outputs, source-self and target-self CLS64; coupled outputs never used',
 'LMMD_implementation':'byte-exact prior snapshot MLUDA mmd.py and Weight.py; 5 RBF kernels,mul2,detached bandwidth and detached target soft class weights',
 'Eq9':'byte-exact audited B1 Pairing kernel, entropy<=0.5ln12,epoch1 fallback,online next-iteration cache+endepoch full eval refresh;private RNG sampling with replacement;empty class native random fallback;retain all source CE samples',
 'initialization':'A/C original backbone tensor hashes identical per seed; build unused original teacher ONLY to preserve historical CPU initialization stream, never forward/update; C new scorer independent RNG seed+41000; no target-driven initialization',
 'streams':'source/candidate pools,order,augmentation,loader/Python/NumPy/CPU RNG same B1; A/C CUDA RNG audited against historical frozen C+CE+Eq9 (no EMA). Actual selected targets allowed to differ by own predictions.',
 'evaluation':'unscored native10287 and full10317 predictions+checkpoint every10 epochs; target GT scored only OFFLINE after training completed; fixed200 FORMAL; target-best diagnostic only, AA/Kappa same OA-best epoch',
 'historical_controls':'CE+LMMD A/C lacks Eq9 and source-val loader iteration differs; C+CE+Eq9 exact parent excludes LMMD; B1+C has native BN and full BiDA objectives. Report these scope differences, never reuse as current A/C.',
 'no_Pavia_changes':True,'no_target_tuning':True,'no_automatic_retries_sweeps_or_followups':True}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def prepare():
 OUT.mkdir(exist_ok=False);dump(OUT/'frozen_plan.json',PLAN)
 shutil.copytree(PRIOR/'runtime_snapshot',OUT/'runtime_snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (OUT/'frozen_official_helper.py').write_text((PRIOR/'frozen_official_helper.py').read_text().replace(str(PRIOR/'runtime_snapshot'),str(OUT/'runtime_snapshot')))
 for name in ('control_snapshot.py','pairing_kernel.py','normalization.py','relation_176.py'):shutil.copy2(PRIOR/name,OUT/name)
 (OUT/'LMMD').mkdir()
 for name in ('mmd.py','Weight.py'):shutil.copy2(OUT.parent/'pavia_backbone_adaptation_v1/snapshot/MLUDA'/name,OUT/'LMMD'/name)
 dump(OUT/'history_audit.json',{'exact_full_configuration_matches':[], 'reuse':False,
 'CE_LMMD':'cross_benchmark_lmmd_v1/hyrank A/C 3seeds200, no Eq9',
 'CE_Eq9':'hyrank_ours_eq9_two_tracks_v1 frozen_ours 3seeds200, C only,noLMMD',
 'B1_C':'same prior bida_full_C, different BN/teacher/coupled/objectives',
 'LMMD_sha256':{name:sha(OUT/'LMMD'/name) for name in ('mmd.py','Weight.py')}})
 originals=json.loads((PRIOR/'original_files_manifest.json').read_text())
 for p in (HIST/'frozen_plan.json',HIST/'snapshot/common.py',HIST/'snapshot/train.py',HIST/'hyrank/summary.json',PRIOR/'summary.json'):
  if p.exists():originals[str(p)]=sha(p)
 dump(OUT/'original_files_manifest.json',originals)
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+list((OUT/'runtime_snapshot').rglob('*.py'))+list(OUT.glob('*.py'))+list((OUT/'LMMD').glob('*.py'))})
 link=ROOT/'results'/OUT.name;assert not link.exists();link.symlink_to(OUT,target_is_directory=True)
def verify():
 for name in ('original_files_manifest.json','code_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p
def environment():
 c=module('joint_lmmd_eq9_helper',OUT/'frozen_official_helper.py')
 nodes=[n for n in ast.parse((OUT/'control_snapshot.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
 ns={'c':c,'ROOT':ROOT,'OUT':OUT,'SEEDS':SEEDS,'ast':ast,'hashlib':hashlib,'json':json,'pickle':__import__('pickle'),
 'random':random,'sys':sys,'Path':Path,'np':np,'torch':torch,'F':torch.nn.functional}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'control_snapshot.py'),'exec'),ns)
 return c,ns

def build(c,args,arm):
 m,unused_teacher,initial=c.build(args,'A');before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
 module('joint_lmmd_eq9_normalization',OUT/'normalization.py').install(m,'fixed_mixture')
 if arm=='C':module('joint_lmmd_eq9_relation',OUT/'relation_176.py').install(m,'C',args.seed)
 assert torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items() if not k.startswith('conv_a.'))
 return m,unused_teacher,initial

def shared_hash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():
  if not k.startswith(('conv_a.','relation_scores.')):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def lmmd():
 sys.modules['Weight']=module('Weight',OUT/'LMMD/Weight.py');value=module('joint_eq9_native_lmmd',OUT/'LMMD/mmd.py')
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;return value
def geometry():return module('prior_geometry',OUT.parent/'bida_hyrank_paper_repro_v4/experiment_code/study.py')
if __name__=='__main__':prepare()
