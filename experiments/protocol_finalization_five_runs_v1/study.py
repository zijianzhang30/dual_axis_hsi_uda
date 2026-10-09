import os,json,hashlib,importlib.util,shutil
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');TOP=B/'protocol_finalization_five_runs_v1'
OBJECTIVE=os.environ.get('PROTOCOL_OBJECTIVE','LMMD');assert OBJECTIVE in ('CE','LMMD');OUT=TOP/OBJECTIVE
CONTROL=B/'hyrank_joint_lmmd_eq9_v1';SEEDS=(2101,2102);ARMS=('C',)
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
parent=module('frozen_eq9_off_protocol',B/'hyrank_lmmd_eq9_off_2100_v1/experiment_code/study.py')
BASE=parent.BASE;PRIOR=parent.PRIOR;random=parent.random;sha=parent.sha;build=parent.build;shared_hash=parent.shared_hash;environment=parent.environment;lmmd=parent.lmmd;geometry=parent.geometry
PLAN=dict(parent.PLAN);PLAN.update({'seeds':[2100,2101,2102],'objective':'source CE ONLY' if OBJECTIVE=='CE' else 'CE+original LMMD schedule .3 ramp/200','Eq9':'DISABLED; same shadow cache computation/RNG schedule as completed seed2100; no cache used for selection','single_factor':'complete fixed seeds2101/2102; same recipe as existing Eq9-off seed2100','controls':'same-objective Eq9-off seed2100 reused','new_training_count':2,'no_source_val_calls':True})
if OBJECTIVE=='CE':PLAN.update(LMMD_schedule='disabled',LMMD_features='unused',LMMD_implementation='not called')
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def verify():
 parent.verify()
 for name in ('code_manifest.json','reuse_manifest.json'):
  for p,h in json.loads((TOP/name).read_text()).items():assert sha(p)==h,p

def prepare():
 TOP.mkdir(exist_ok=False)
 for name in ('CE','LMMD','Houston_lr001'):(TOP/name).mkdir()
 for name in ('CE','LMMD'):shutil.copy2(B/'hyrank_lmmd_eq9_off_2100_v1/pairing_kernel.py',TOP/name/'pairing_kernel.py')
 reuse={}
 for name in ('hyrank_ce_eq9_off_2100_v1','hyrank_lmmd_eq9_off_2100_v1'):
  f=B/name/'C_2100';m=json.loads((f/'frozen_manifest.json').read_text());assert m['completed_epochs']==200
  for p in f.glob('*.json*'):reuse[str(p)]=sha(p)
  for r in m['evaluations']:
   for kind in ('native','full'):reuse[r[kind+'_prediction_path']]=r[kind+'_sha256']
 for seed in (2101,2102):
  f=CONTROL/('C_%d'%seed);rows=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(rows)==200 and all(r['steps']==80 for r in rows)
  for p in (f/'config.json',f/'history.jsonl'):reuse[str(p)]=sha(p)
 f=B/'cross_benchmark_lmmd_v1/houston/C_LMMD_2100';m=json.loads((f/'frozen_manifest.json').read_text());assert m['checkpoints'][-1]['epoch']==200
 for p in (f/'config.json',f/'history.jsonl',f/'frozen_manifest.json'):reuse[str(p)]=sha(p)
 for r in m['checkpoints']:reuse[r['path']]=r['sha256']
 code=TOP/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(TOP/'reuse_manifest.json',reuse);dump(TOP/'code_manifest.json',{str(p):sha(p) for p in code.glob('*.py')})
 dump(TOP/'frozen_plan.json',{'HyRANK':'complete CE and CE+LMMD withoutEq9 seeds2101/2102,4new runs;existing seed2100 exact controls reused','Houston':'CE+LMMD no adapter noEq9 seed2100,200epochs,ONLY lr .01 -> .001,1new run','max_new_trainings':5,'no_parameter_search':True,'no_Pavia':True,'no_new_structures':True,'report':'perseedtargetbest grid10:10:200 OA-selected AA/Kappa,mean sample std;fixed200 separately;native/full distinct','no_auto_followups':True})
 dump(TOP/'history_audit.json',{'HyRANK_complete_matching_seeds2101_2102_not_found':True,'historical_R0_LMMD_sourceval_RNG_differs_cannot_substitute':True,'HyRANK2100_CE_LMMD_reused':True,'Houston_lr001_complete_matching_not_found':True})
 (ROOT/'results'/TOP.name).symlink_to(TOP,target_is_directory=True)
if __name__=='__main__':prepare()
