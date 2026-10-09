import importlib.util,json,shutil,hashlib
from pathlib import Path
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_adapter_lmmd_formal_v1')
HIST=OUT.parent/'cross_benchmark_lmmd_v1';SEEDS=(2100,2101,2102);ARMS=('R1','R2')
PLAN={'dataset':'HyRANK Dioni->Loukia','seeds':list(SEEDS),'epochs':200,'R0':'reuse historical cross_benchmark_lmmd_v1/hyrank/C_LMMD_SEED; never retrain',
 'R1':'shared bias-free64->16->GELU->64 adapter on all3 FFN blocks, zero up initialization;6144 extra total',
 'R2':'separate source/target SAME per-path16 width;12288 extra total;source/target initialization identical',
 'capacity_limitation':'same path width, R2 total parameter budget is doubled; report honestly, no dummy params, cannot fully exclude total-capacity confound',
 'base':'unchanged historical C tokenizer + FullJointBN.19 +3-layer self paths; unused historical coupled products do not enter loss or adapters',
 'recipe':'EXACT historical CE+LMMD,noEq9,noCenterLoss initialization,noadditionalMMD, distillation, teacher forward/update or EMA',
 'LMMD':'byte-exact old implementation on self CLS64;weight .3*(2/(1+exp(-10*epoch/200))-1)',
 'optimizer':'SGD .01 momentum0 weight_decay0 no scheduler;13patch,176bands,batch128,80updates/epoch',
 'sampler':'original independently shuffled source and target; historical loader/source95% split and original source-val iterator schedule retained',
 'source_val':'epoch1 and every10, same loader calls/RNG as historical; SOURCE adapter explicit in eval, target adapter never used for source validation',
 'inference':'target adapter explicit,target input only,running BN,no source scene needed',
 'initialization':'new adapters independent RNG seed+73000; underlying C tensors and model/dropout/data RNG unchanged',
 'evaluation':'after all200, freeze predictions10:10:200; source-val checkpoint1 excluded from target-best grid for ALL groups; native/full separately; both fixed200 and target-best,AA/Kappa same OA-selected epoch',
 'GT':'target class values zero before any training iterator; target GT only offline scoring after output freezing',
 'report':'R2-R0 and R2-R1 paired3seed,perclass,collapse,zeroRecall,source-val,targetLMMDgradients,params,time',
 'new_training_count':6,'no_Houston_or_Pavia_changes':True,'no_parameter_sweeps_or_auto_followups':True}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare():
 OUT.mkdir(exist_ok=False);dump(OUT/'frozen_plan.json',PLAN);snap=OUT/'snapshot';snap.mkdir()
 oldhelper=ROOT/'experiments/official_bench_v1/common.py'
 runtime=snap/'BiDA';shutil.copytree('/home/zhangzj26/IEEE_TCSVT_BiDA',runtime,ignore=shutil.ignore_patterns('.git','experiments','results','__pycache__','*.pyc','*.pth','*.mat'))
 for src,name in [(ROOT/'experiments/bn_stabilization_v1/normalization.py','normalization.py'),(ROOT/'experiments/relation_tokenizer_v1/relation.py','relation.py'),(ROOT/'experiments/bida_self_dropout/train.py','metrics_helper.py')]:shutil.copy2(src,snap/name)
 helper=oldhelper.read_text().replace('ROOT=Path(__file__).resolve().parents[2]',"ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')")
 helper=helper.replace("BIDA=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')",'BIDA=Path('+repr(str(runtime))+')')
 helper=helper.replace("ROOT/'experiments/bida_self_dropout/train.py'",'Path('+repr(str(snap/'metrics_helper.py'))+')')
 helper=helper.replace("ROOT/'experiments/bn_stabilization_v1/normalization.py'",'Path('+repr(str(snap/'normalization.py'))+')')
 helper=helper.replace("ROOT/'experiments/relation_tokenizer_v1/relation.py'",'Path('+repr(str(snap/'relation.py'))+')')
 (snap/'official_common.py').write_text(helper)
 old=(HIST/'snapshot/common.py').read_text().replace("ROOT/'experiments/official_bench_v1/common.py'",'Path('+repr(str(snap/'official_common.py'))+')')
 old=old.replace("p=RESULTS/'pavia_backbone_adaptation_v1/snapshot/MLUDA'",'p=Path('+repr(str(snap/'LMMD'))+')')
 (snap/'legacy_common.py').write_text(old)
 (snap/'LMMD').mkdir()
 for name in ('mmd.py','Weight.py'):shutil.copy2(OUT.parent/'pavia_backbone_adaptation_v1/snapshot/MLUDA'/name,snap/'LMMD'/name)
 shutil.copy2(Path(__file__).parent/'adapter.py',OUT/'adapter.py')
 history={}
 for seed in SEEDS:
  f=HIST/'hyrank'/('C_LMMD_%d'%seed);m=json.loads((f/'frozen_manifest.json').read_text());rows=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()]
  assert len(rows)==200 and all(r['steps']==80 for r in rows) and m['checkpoints'][-1]['epoch']==200
  for name in ('config.json','history.jsonl','frozen_manifest.json'):history[str(f/name)]=sha(f/name)
  for r in m['checkpoints']:assert sha(r['path'])==r['sha256'];history[r['path']]=r['sha256']
  history[m['prediction']['path']]=m['prediction']['sha256']
 dump(OUT/'historical_result_manifest.json',history)
 code=OUT/'experiment_code';code.mkdir()
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
 dump(OUT/'code_manifest.json',{str(p):sha(p) for p in list(code.glob('*.py'))+list(snap.rglob('*.py'))+[OUT/'adapter.py']})
 (ROOT/'results'/OUT.name).symlink_to(OUT,target_is_directory=True)

def environment():return module('formal_adapter_legacy',OUT/'snapshot/legacy_common.py')
def verify():
 # Historical implementation and all preserved controls must still match their recorded provenance.
 environment().verify_files()
 for name in ('historical_result_manifest.json','code_manifest.json'):
  for p,h in json.loads((OUT/name).read_text()).items():assert sha(p)==h,p
if __name__=='__main__':prepare()
