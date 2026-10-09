import ast,hashlib,importlib.util,json,random,shutil,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_ours_eq9_two_tracks_v1')
B1=OUT.parent/'bida_hyrank_eq9_pairing_v1'
BASE=OUT.parent/'hyrank_A_no_extra_val_v1'
SEEDS=(2100,2101,2102)
TRACKS=('frozen_ours','bida_full_C')
PLAN={'seeds':list(SEEDS),'epochs':200,'patch':13,'bands':176,'batch':128,'steps_per_epoch':80,
 'optimizer':'SGD lr.01 momentum0 weight_decay0, no scheduler',
 'frozen_ours':'Full Joint BN momentum.19 + C relation tokenizer + original self-path Transformer; ONLY source CE; no teacher forward/update, coupled loss/distillation/MMD. Original forward retains unused coupled computations for historical implementation equivalence.',
 'bida_full_C':'ONLY replace student AND teacher original tokenizer by C; native sequential CNN BN momentum.1, complete released BiDA loss/EMA/coupled logic unchanged, including original self-self MMD bug.',
 'initialization':'Build native student and independently initialized native teacher in original order, then install C scorer with isolated CPU RNG seed+41000; all backbone tensors stay native. Teacher C scorer uses the same isolated seed; original teacher backbone independent init preserved.',
 'pairing':'Exact frozen B1 kernel/policies: native source batches unchanged; epoch1 random fallback; entropy<=.5ln12; cached target same-pseudoclass sampling with replacement; empty class native random fallback; private sampler/augmentation RNG; full eval-only cache refresh every epoch plus online next-iteration updates.',
 'pairing_limitation':'No coupled supervision in frozen_ours; class pairing affects the target distribution used by Joint BN, not source-target cross attention losses.',
 'data':'Original B1 source sampling, candidate mask, normalization, augmentation, loaders/worker/shared-generator setup; native unequal final-batch skip preserved.',
 'evaluation':'Every10 fixed schedule. Native10287 and full10317 separate. Retain fixed200 and each seed highest OA after ALL200 epochs; AA/Kappa from SAME OA-selected epoch; report all three seeds, no best-seed cherry picking.',
 'GT':'Target values zero in training/pairing/cache; only native candidate mask inherited, GT used in evaluation/offline diagnostics; no target-driven changes to training/config.',
 'random_streams':'Same source/candidate data/augmentation and CPU/loader RNG as B0/B1; full_C also asserts CUDA RNG identical; frozen_ours no EMA forward so its CUDA dropout stream necessarily differs from full BiDA, explicitly disclosed.',
 'not_included':'MMD fix, new losses, Pavia changes, new architectures, hyperparameter search',
 'no_automatic_followups':True}
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
 shutil.copytree(B1/'runtime_snapshot',OUT/'runtime_snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 helper=(B1/'frozen_official_helper.py').read_text().replace(str(B1/'runtime_snapshot'),str(OUT/'runtime_snapshot'))
 (OUT/'frozen_official_helper.py').write_text(helper);shutil.copy2(B1/'control_snapshot.py',OUT/'control_snapshot.py')
 shutil.copy2(B1/'experiment_code/pairing.py',OUT/'pairing_kernel.py')
 shutil.copy2(ROOT/'experiments/bn_stabilization_v1/normalization.py',OUT/'normalization.py')
 original=(ROOT/'experiments/relation_tokenizer_v1/relation.py').read_text()
 derived=original
 for a,b in [('bands=48','bands=176'),('(1, 48)','(1, 176)'),('B×1×48×H×W','B×1×176×H×W')]:
  assert derived.count(a)==1;derived=derived.replace(a,b)
 (OUT/'relation_176.py').write_text(derived)
 dump(OUT/'original_files_manifest.json',json.loads((B1/'original_files_manifest.json').read_text()))
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+list((OUT/'runtime_snapshot').rglob('*.py'))+list(OUT.glob('*.py'))})
 link=ROOT/'results'/OUT.name;assert not link.exists();link.symlink_to(OUT,target_is_directory=True)
def verify():
 for manifest in ('original_files_manifest.json','code_manifest.json'):
  for p,h in json.loads((OUT/manifest).read_text()).items():assert sha(p)==h,p
def environment():
 c=module('ours_eq9_release_helper',OUT/'frozen_official_helper.py')
 nodes=[n for n in ast.parse((OUT/'control_snapshot.py').read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
 ns={'c':c,'ROOT':ROOT,'OUT':OUT,'SEEDS':SEEDS,'ast':ast,'hashlib':hashlib,'json':json,'pickle':__import__('pickle'),
 'random':random,'sys':sys,'Path':Path,'np':np,'torch':torch,'F':torch.nn.functional}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OUT/'control_snapshot.py'),'exec'),ns)
 return c,ns

def build(c,args,track):
 student,teacher,initial=c.build(args,'A')
 orig={k:v.clone() for k,v in student.state_dict().items()};orig_t={k:v.clone() for k,v in teacher.state_dict().items()}
 rng=torch.get_rng_state().clone();adapter=module('ours_eq9_relation',OUT/'relation_176.py')
 if track=='frozen_ours':module('ours_eq9_joint',OUT/'normalization.py').install(student,'fixed_mixture')
 adapter.install(student,'C',args.seed)
 if track=='bida_full_C':adapter.install(teacher,'C',args.seed)
 assert torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(v,student.state_dict()[k]) for k,v in orig.items() if not k.startswith('conv_a.'))
 assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in orig_t.items() if not k.startswith('conv_a.'))
 if track=='bida_full_C':
  assert list(student.state_dict())==list(teacher.state_dict())
  # The installed module did not exist when native get_model detached teacher parameters.
  for p in teacher.parameters():p.detach_()
 return student,teacher,initial

def geometry():return module('prior_geometry',B1.parent/'bida_hyrank_paper_repro_v4/experiment_code/study.py')
if __name__=='__main__':prepare()
