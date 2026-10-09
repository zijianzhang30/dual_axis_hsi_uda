import importlib.util,json,shutil,hashlib
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_lmmd_eq9_off_2100_v1')
CONTROL=OUT.parent/'hyrank_joint_lmmd_eq9_v1';SEEDS=(2100,);ARMS=('C',)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
parent=module('eq9_off_frozen_control',CONTROL/'experiment_code/study.py')
BASE=parent.BASE;PRIOR=parent.PRIOR;random=parent.random
sha=parent.sha;build=parent.build;shared_hash=parent.shared_hash;environment=parent.environment;lmmd=parent.lmmd;geometry=parent.geometry
PLAN=dict(parent.PLAN)
PLAN.update({'arms':['C'],'seeds':[2100],'max_new_training_runs':1,'single_factor':'Pairing enabled=False: always use original shuffled target batch, no class-matched replacement',
 'Eq9':'DISABLED. Existing online/eval-only cache computations retained for identical compute/BN/RNG accounting; cache values NEVER select training inputs.',
 'source_val_calls':'none, EXACT current unified recipe, not legacy source-val schedule',
 'controls':'reuse completed unified C+CE+LMMD+Eq9 seed2100 same200 epochs',
 'objective':'unchanged CE+LMMD, original fixed ramp denominator200',
 'report':'offline after200: fixed200 and target-best separately, same selected OA epoch for AA/Kappa; native/full separately; C7/C8 trajectory and paired at SAME epochs',
 'no_adapters':True,'no_automatic_followups':True})
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def verify():
 parent.verify()
 for p,h in json.loads((OUT/'code_manifest.json').read_text()).items():assert sha(p)==h,p

def prepare():
 OUT.mkdir(exist_ok=False);dump(OUT/'frozen_plan.json',PLAN)
 shutil.copy2(CONTROL/'pairing_kernel.py',OUT/'pairing_kernel.py')
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+[OUT/'pairing_kernel.py']})
 (ROOT/'results'/OUT.name).symlink_to(OUT,target_is_directory=True)
 dump(OUT/'history_audit.json',{'exact_completed_matching_result':False,'legacy_C_LMMD_different_source_val_schedule':True,
 'historical_controls_preserved':str(CONTROL/'C_2100'),'one_new_train_only':True})
if __name__=='__main__':prepare()
