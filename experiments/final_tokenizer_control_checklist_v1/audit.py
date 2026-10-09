"""No model forwards or training: config, stream and saved artifact audit only."""
from pathlib import Path
import csv,json,hashlib,numpy as np
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');O=B/'final_tokenizer_control_checklist_v1';O.mkdir(exist_ok=True);assert not (O/'audit.json').exists(), 'Refuse completed audit overwrite'
SEEDS=(2100,2101,2102);sources={};checked=[];cells=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):
 p=Path(p);sources[str(p)]=sha(p);return json.loads(p.read_text())
def history(p):
 p=Path(p);sources[str(p)]=sha(p);return [json.loads(x) for x in p.read_text().splitlines()]
def artifact(folder):
 m=read(folder/'frozen_manifest.json');assert m['checkpoints'][-1]['epoch'] in (100,200)
 for r in m['checkpoints']+[m['prediction']]:assert sha(r['path'])==r['sha256'];checked.append({'path':r['path'],'sha256':r['sha256']})
 return m
def aggregate(v):return {'values':v,'mean':float(np.mean(v)),'std_sample':float(np.std(v,ddof=1))}
def pair(benchmark,objective,protocol,af,cf,epochs,keys,config_hash,data_key,summary_pair,targetbest):
 checks=[]
 for seed in SEEDS:
  a=af(seed);c=cf(seed);ac=read(a/'config.json');cc=read(c/'config.json');ah=history(a/'history.jsonl');ch=history(c/'history.jsonl')
  assert len(ah)==len(ch)==epochs and all(x['epoch']==y['epoch']==i+1 for i,(x,y) in enumerate(zip(ah,ch)))
  for k in keys:assert all(x[k]==y[k] for x,y in zip(ah,ch)),(benchmark,objective,seed,k)
  assert ac[config_hash]==cc[config_hash],(benchmark,objective,seed,'non-tokenizer init')
  if data_key:assert ac[data_key]==cc[data_key]
  for k in ('epochs','patch','batch','lr','depth','dropout','BN','tokens','dim','settings','data_protocol','normalized_cube_sha256','optimizer','data','all_cells_forward','weights','objective','enabled_terms','BN','alignment_features'):
   if k in ac and k in cc:
    av,cv=ac[k],cc[k]
    if k=='settings':av={kk:vv for kk,vv in av.items() if kk!='device'};cv={kk:vv for kk,vv in cv.items() if kk!='device'}
    assert av==cv,(benchmark,objective,seed,k)
  am=artifact(a);cm=artifact(c);checks.append({'seed':seed,'A':str(a),'C':str(c),'epochs':epochs,'actual_steps_A':sum(x['steps'] for x in ah),'actual_steps_C':sum(x['steps'] for x in ch),'same_init_non_tokenizer_hash':ac[config_hash],'same_data_config_hashes':True,'same_every_epoch_stream_keys':keys,'all_checkpoints_and_final_predictions_hash_verified':True,'physical_GPU_device_ids_excluded_from_recipe_comparison':True})
 cells.append({'benchmark':benchmark,'objective':objective,'protocol':protocol,'status':'STRICT_MATCH_COMPLETE_3_SEEDS','epochs':epochs,'paired_fixed_endpoint':summary_pair,'target_best_status':targetbest,'checks':checks})
# Existing verified summaries, fixed endpoints. No target-best substituted for them.
h=read(B/'tokenizer_mechanism_v2/houston/summary.json')
hlm=read(B/'cross_benchmark_lmmd_v1/houston/summary.json')
pl=read(B/'tokenizer_mechanism_pavia_l2_fixed_v1/pavia/summary.json');pm=read(B/'pavia_lmmd_tokenizer_v1/summary.json')
def metric_pair(A,C,oa='oa_percent',aa='aa_percent',kap='kappa'):
 result={}
 for key,label in ((oa,'OA'),(aa,'AA'),(kap,'Kappa')):
  av=A[key].get('values');cv=C[key].get('values')
  if av is not None and cv is not None:result[label]=aggregate([y-x for x,y in zip(av,cv)])
 return result
pair('Houston','CE','normband48,patch13,bs128,SGD.01/0/0,200,sourceval1/every10',lambda s:B/'tokenizer_mechanism_v2/houston'/('A_%d'%s),lambda s:B/'tokenizer_mechanism_v2/houston'/('C_%d'%s),200,['batch_stream_sha256','original_rng_stream_sha256'],'initial_original_model_sha256','indices_sha256',metric_pair(h['aggregate']['A'],h['aggregate']['C']),'C target-best exists; A same20 checkpoints available but unscored; paired-best missing only evaluation')
# Derive exact paired fixed metrics from original per-seed rows.
r={}
for x in hlm['rows']:
 if x['eval_protocol']=='native' and x['objective']=='LMMD':r[(x['arm'],x['seed'])]=x
mp={k:aggregate([r[('C',s)][rk]-r[('A',s)][rk] for s in SEEDS]) for k,rk in [('OA','oa_percent'),('AA','aa_percent'),('Kappa','kappa')]}
pair('Houston','CE+LMMD','same Houston protocol;LMMD.3 ramp/200,noEq9',lambda s:B/'cross_benchmark_lmmd_v1/houston'/('A_LMMD_%d'%s),lambda s:B/'cross_benchmark_lmmd_v1/houston'/('C_LMMD_%d'%s),200,['batch_stream_sha256','model_rng_sha256','BN_batches'],'shared_initial_hash','indices_sha256',mp,'C target-best exists; A same20 checkpoints available but unscored; paired-best missing only evaluation')
pair('Pavia','CE','REPAIRED normalized L2,102bands,patch13,bs128,SGD.01/0/0,200',lambda s:B/'tokenizer_mechanism_pavia_l2_fixed_v1/pavia'/('A_%d'%s),lambda s:B/'tokenizer_mechanism_pavia_l2_fixed_v1/pavia'/('C_%d'%s),200,['batch_stream_sha256','original_rng_stream_sha256'],'initial_original_model_sha256','indices_sha256',metric_pair(pl['aggregate']['A'],pl['aggregate']['C']),'Both same20 checkpoints present; paired target-best not scored in saved summary; evaluation only gap')
pair('Pavia','CE+LMMD','P2 zscore+ILDA,180/class,patch11,bs32,SGD.001/.9/5e-4 recreated each epoch,3views,100',lambda s:B/'pavia_lmmd_tokenizer_v1'/('A_LMMD_%d'%s),lambda s:B/'pavia_lmmd_tokenizer_v1'/('C_LMMD_%d'%s),100,['plan_sha256','all_input_views_sha256','model_rng_sha256','BN_num_batches'],'initial_shared_model_sha256',None,metric_pair(pm['aggregate']['A_LMMD'],pm['aggregate']['C_LMMD'],'OA','AA','Kappa'),'Both saved epoch10:10:100 (+1 excluded) checkpoints present; target-best unscored; evaluation only gap')
# Latest HyRANK exact noEq9 recipe: all3 C completed, no matching A studies.
latest=read(B/'protocol_finalization_five_runs_v1/summary.json')
for objective in ('CE','CE+LMMD'):
 name='CE' if objective=='CE' else 'LMMD';checks=[]
 for seed in SEEDS:
  folder=(B/('hyrank_ce_eq9_off_2100_v1' if name=='CE' else 'hyrank_lmmd_eq9_off_2100_v1')/'C_2100') if seed==2100 else B/'protocol_finalization_five_runs_v1'/name/('C_%d'%seed)
  cfg=read(folder/'config.json');hh=history(folder/'history.jsonl');m=read(folder/'frozen_manifest.json');assert len(hh)==200 and m['completed_epochs']==200 and all(x['steps']==80 and not x['Eq9_enabled'] for x in hh)
  checks.append({'seed':seed,'C':str(folder),'A':None,'source_val_calls':'none','Eq9_enabled':False,'shadow_cache_eval_only_preserved_for_RNG_and_compute':True,'unused_make_loss_initialization':'preserved','completed_epochs':200})
 cells.append({'benchmark':'HyRANK','objective':objective,'protocol':'LATEST noEq9,176bands,patch13,bs128,200,no source-val calls,unused loss init+shadow eval cache retained','status':'C_COMPLETE_3_SEEDS_A_MISSING_ALL_3','epochs':200,'paired_fixed_endpoint':None,'target_best_status':'C scored on grid10:10:200; A not trained, cannot infer paired delta','checks':checks})
# Historical HyRANK matched A/C still usable as a separate recipe, never with latest C.
old=read(B/'official_bench_v1/summary.json');a=old['aggregate']['B'];c=old['aggregate']['C'];oldrows={}
for x in old['rows']:
 if x['arm'] in ('B','C'):oldrows[(x['arm'],x['seed'])]=x
oldpair={k:aggregate([oldrows[('C',s)][rk]-oldrows[('B',s)][rk] for s in SEEDS]) for k,rk in [('OA','oa_percent'),('AA','aa_percent'),('Kappa','kappa')]}
pair('HyRANK historical','CE','OLD noEq9; sourceval1/every10,NO unused center-loss init/shadow cache; B means original tokenizer A-self,not completeBiDA',lambda s:B/'official_bench_v1'/('hyrank_B_%d'%s),lambda s:B/'official_bench_v1'/('hyrank_C_%d'%s),200,['batch_stream_sha256','rng_stream_sha256'],'initial_common_model_sha256','data_protocol',oldpair,'Neither A/C paired best scored here; checkpoints exist; separate OLD recipe only')
o=read(B/'cross_benchmark_lmmd_v1/hyrank/summary.json');rs={}
for x in o['rows']:
 if x['eval_protocol']=='native' and x['objective']=='LMMD':rs[(x['arm'],x['seed'])]=x
mp={k:aggregate([rs[('C',s)][rk]-rs[('A',s)][rk] for s in SEEDS]) for k,rk in [('OA','oa_percent'),('AA','aa_percent'),('Kappa','kappa')]}
pair('HyRANK historical','CE+LMMD','OLD noEq9; sourceval1/every10,NO unused center-loss init/shadow cache; original A-self vs C-self',lambda s:B/'cross_benchmark_lmmd_v1/hyrank'/('A_LMMD_%d'%s),lambda s:B/'cross_benchmark_lmmd_v1/hyrank'/('C_LMMD_%d'%s),200,['batch_stream_sha256','model_rng_sha256','BN_batches'],'shared_initial_hash','indices_sha256',mp,'C R0 best rescored; A checkpoint grid exists but unscored; separate OLD recipe only')
result={'read_only':True,'no_training_started':True,'no_model_forward_or_new_evaluation':True,'cells':cells,'missing_latest_HyRANK_trainings_if_later_authorized':['A+CE seed2100','A+CE seed2101','A+CE seed2102','A+CE+LMMD seed2100','A+CE+LMMD seed2101','A+CE+LMMD seed2102'],'Pavia_same_protocol_2x2_limitation':'CE L2fixed200 vs LMMD P2fixed100; valid within-cell A/C only,not isolated objective interaction. P2 C_CE3 completed but matching A_CE P2 missing. Do not replace it with L2 A_CE.','excluded':['Pavia raw-input diagnostic','Full BiDA ordinary MMD/coupled/distillation/EMA or BiDA+C','Adapters R1/R2','Houston lr.001 single-seed C-only diagnostic'],'source_manifest':sources,'checked_result_artifacts':checked,'selection_policy':'User prefers per-seed OA-best then mean sample std,AA/Kappa same epoch. Until paired-best is scored, only present fixed-endpoint evidence with explicit endpoint label; never substitute or mix endpoints.'}
(O/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
rows=[]
for x in cells:
 p=x['paired_fixed_endpoint'];rows.append({'benchmark':x['benchmark'],'objective':x['objective'],'protocol':x['protocol'],'status':x['status'],'fixed_epoch':x['epochs'],'C_minus_A_OA_pp':p['OA']['mean'] if p else None,'C_minus_A_AA_pp':p['AA']['mean'] if p else None,'OA_positive_seeds':sum(v>0 for v in p['OA']['values']) if p else None,'AA_positive_seeds':sum(v>0 for v in p['AA']['values']) if p else None,'target_best_status':x['target_best_status']})
with (O/'checklist.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(sha(p)==h for p,h in sources.items())
for r in rows:print(r)
print('CHECKLIST_SAVED',O)
