"""Frozen source-coverage-preserving reconstruction of paper Eq.9, release losses intact."""
import ast, hashlib, importlib.util, json, os, random, shutil, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/bida_hyrank_eq9_pairing_v1')
PRIOR=OUT.parent/'bida_hyrank_paper_repro_v4'
BASE=OUT.parent/'hyrank_A_no_extra_val_v1'
SEEDS=(2100,2101,2102)
PLAN={'name':'B1: release BiDA + source-coverage-preserving Eq9 class pairing reconstruction',
 'official_commit':'66a92b542ffa1df3c8b628af3d8bd001b799f1f6','seeds':list(SEEDS),'epochs':200,
 'B0':'reuse completed native-equivalent original release 3 seeds; no retraining',
 'B1_only_factor':'replace target rows by entropy-eligible cached same-pseudo-class candidates for the UNCHANGED original source batch',
 'threshold':'per-sample H(softmax(student target self logits)) <= 0.5*ln(12)',
 'first_epoch':'original random target pairing throughout epoch1; online predictions cached but not used until epoch2',
 'cache':'indexed by native target dataset id; update after every student main forward using detached target-self logits; duplicate ids averaged within batch; refresh all target ids once at end of each epoch using eval/running BN and no_grad; restore every module mode and all global RNG states',
 'cache_eval':'private nonshuffled no-augmentation loader, separate torch.Generator; no source-loader generator access; target class values replaced by zeros',
 'sampling':'uniform with replacement from all cached eligible targets of the source true class; private RandomState(seed+90219), independent private augmentation RandomState(seed+90220)',
 'empty_class_or_no_eligible':'keep the original random target row; label it unmatched; no sample removal, no loss masking, no source resampling',
 'batch_refill':'same-class sampling with replacement, unmatched slots use original candidate row; retain native skip of unequal last batch, exactly80 optimizer steps per epoch',
 'augmentation':'retain original candidate augmentation when selected id equals original same-row id; otherwise apply EXACT native augmentation using a private NumPy RNG, restore global state; student/EMA see the same final input',
 'unchanged':'original CNN/tokenizer/Transformer, sequential BN momentum.1, source CE/all loss formulas and weights, original self-self MMD bug, MMD epoch>100 schedule, EMA init/update/modes, SGD .01 no momentum/weight decay, 13patch,128batch, original source sample split and source augmentation/order',
 'evaluation':'fixed200 primary; OA-best of existing every10 evaluations diagnostic only; native10287 and full10317 saved separately',
 'GT':'target labels only fixed native geometry and offline scoring; never pseudo-label cache, selector, threshold, losses or follow-up decisions',
 'paper_ambiguity':'paper does not specify first iteration, empty classes, refill or cache maintenance; these policies are predeclared reconstruction choices, not author-confirmed implementation',
 'refresh_cost':'extra eval-only prediction sweep each epoch except reuse native validation logits on every10 epoch; no extra TRAIN BN/Transformer forward',
 'independent_MMD_repair':'continues separately; do not combine', 'no_Ours_training':True,'no_parameter_sweep':True,'no_automatic_followup':True}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def prepare():
 OUT.mkdir(exist_ok=False);dump(OUT/'frozen_plan.json',PLAN)
 runtime=OUT/'runtime_snapshot'
 shutil.copytree(PRIOR/'runtime_snapshot',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2(PRIOR/'released_train_pipeline.py',runtime/'train_pipeline.py')
 assert 'out_x_fusion_src, out_x_fusion_src' in (runtime/'train_pipeline.py').read_text()
 helper=(PRIOR/'frozen_official_helper.py').read_text().replace(str(PRIOR/'runtime_snapshot'),str(runtime))
 (OUT/'frozen_official_helper.py').write_text(helper)
 shutil.copy2(PRIOR/'control_snapshot.py',OUT/'control_snapshot.py')
 originals=json.loads((PRIOR/'original_files_manifest.json').read_text())
 dump(OUT/'original_files_manifest.json',originals)
 verify_originals()
 link=ROOT/'results'/OUT.name
 assert not link.exists();link.symlink_to(OUT,target_is_directory=True)
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+list(runtime.rglob('*.py'))+[OUT/'frozen_official_helper.py',OUT/'control_snapshot.py']})
def verify_originals():
 for p,h in json.loads((OUT/'original_files_manifest.json').read_text()).items():assert sha(p)==h,p
def environment():
 c=module('eq9_frozen_official_helper',OUT/'frozen_official_helper.py')
 nodes=[n for n in ast.parse((OUT/'control_snapshot.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
 ns={'c':c,'ROOT':ROOT,'OUT':OUT,'SEEDS':SEEDS,'ast':ast,'hashlib':hashlib,'json':json,'pickle':__import__('pickle'),
 'random':random,'sys':sys,'Path':Path,'np':np,'torch':torch,'F':torch.nn.functional}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'control_snapshot.py'),'exec'),ns)
 return c,ns
# Read-only geometry/evaluation helpers from the separately frozen previous audit.
def geometry():return module('prior_readonly_geometry',PRIOR/'experiment_code/study.py')
if __name__=='__main__':prepare()
