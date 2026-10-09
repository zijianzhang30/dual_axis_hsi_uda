import json,hashlib,importlib.util,shutil
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');RESULTS=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=RESULTS/'houston_shared_adapter_2x2_v1'
HIST=RESULTS/'cross_benchmark_lmmd_v1';SEEDS=(2100,2101,2102);ARMS=('CE1','LMMD1');GROUPS=('CE0','CE1','LMMD0','LMMD1')
PLAN={'dataset':'Houston13 -> Houston18','seeds':list(SEEDS),'epochs':200,'groups':{'CE0':'existing C+source CE','CE1':'C+source CE+shared adapter','LMMD0':'existing C+CE+LMMD','LMMD1':'C+CE+LMMD+shared adapter'},'base':'Full Joint BN .19+C tokenizer+3-layer self Transformer; no Eq9,coupled objective,distillation,EMA,extra MMD','adapter':'exact HyRANK formal R1 bias-free64->16->GELU->64 on each3 FFN self-paths;6144 extra;zero up initialization;private RNG seed+73000','recipe':'unchanged Houston normband48bands,13patch,128batch,SGD .01 momentum0 weight_decay0,no scheduler,18steps/epoch;source-val epoch1/every10 preserved','LMMD':'unchanged .3*(2/(1+exp(-10*epoch/200))-1), existing MLUDA implementation CLS64; no LMMD computation in CE groups','initialization':'same original C initial hash,source/target cohorts,data/augmentation and dropout RNG against both existing controls','evaluation':'no training target scoring; offline same epochs10:10:200; fixed200 AND per-seed target-best with AA/Kappa same selected OA checkpoint','comparisons':['CE1-CE0','LMMD1-LMMD0','LMMD1-CE0','I=(LMMD1-LMMD0)-(CE1-CE0)'],'decision':'report paired OA/AA average,directions,third-seed degradation,class coverage;3seed screening evidence only,not proof of universal effectiveness;positive interaction alone not Go','development_caveat':'Houston/Pavia already used for development; candidate screening,not independent generalization validation','HyRANK':'reuse completed formal R0/R1,NO retraining','Pavia':'DEFERRED; do not auto launch','new_training_count':6,'no_auto_variants_or_followups':True}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment():return module('houston_adapter_frozen_protocol',HIST/'snapshot/common.py')
def baseline(group,seed):return RESULTS/'tokenizer_mechanism_v2/houston'/('C_%d'%seed) if group=='CE0' else HIST/'houston'/('C_LMMD_%d'%seed)
def prepare():
 OUT.mkdir(exist_ok=False);dump(OUT/'frozen_plan.json',PLAN);sources={};audits=[]
 for seed in SEEDS:
  ce=baseline('CE0',seed);lm=baseline('LMMD0',seed);cc=json.loads((ce/'config.json').read_text());lc=json.loads((lm/'config.json').read_text())
  assert cc['initial_full_model_sha256']==lc['complete_initial_hash'] and cc['indices_sha256']==lc['indices_sha256']
  cr=[json.loads(x) for x in (ce/'history.jsonl').read_text().splitlines()];lr=[json.loads(x) for x in (lm/'history.jsonl').read_text().splitlines()]
  assert len(cr)==len(lr)==200
  assert all(a['batch_stream_sha256']==b['batch_stream_sha256'] and a['original_rng_stream_sha256']==b['model_rng_sha256'] and a['steps']==b['steps']==18 for a,b in zip(cr,lr))
  for folder in (ce,lm):
   mf=json.loads((folder/'frozen_manifest.json').read_text())
   for f in ('config.json','history.jsonl','frozen_manifest.json'):sources[str(folder/f)]=sha(folder/f)
   for r in mf['checkpoints']:assert sha(r['path'])==r['sha256'];sources[r['path']]=r['sha256']
   sources[mf['prediction']['path']]=mf['prediction']['sha256']
  audits.append({'seed':seed,'CE_and_LMMD_controls_complete':True,'initial_and_cohort_hashes_match':True,'all200_data_and_dropout_streams_match':True,'CE0':str(ce),'LMMD0':str(lm),'new_adapter_groups_historical_match':False})
 # Existing Houston adapter studies use different center-CLS/readout/prototype modules; no matching shared FFN16 study found.
 hy=RESULTS/'hyrank_adapter_lmmd_formal_v1/summary.json';sources[str(hy)]=sha(hy);dump(OUT/'HyRANK_reused_summary.json',json.loads(hy.read_text()))
 for p in (RESULTS/'tokenizer_mechanism_v2').glob('*.py'):sources[str(p)]=sha(p)
 for p in (RESULTS/'tokenizer_mechanism_v2/snapshot').rglob('*.py'):sources[str(p)]=sha(p)
 sources[str(HIST/'snapshot/common.py')]=sha(HIST/'snapshot/common.py')
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in code.glob('*.py')});dump(OUT/'historical_and_dependency_manifest.json',sources);dump(OUT/'history_audit.json',audits)
 (ROOT/'results'/OUT.name).symlink_to(OUT,target_is_directory=True)
def verify():
 environment().verify_files()
 for name in ('code_manifest.json','historical_and_dependency_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p
if __name__=='__main__':prepare()
