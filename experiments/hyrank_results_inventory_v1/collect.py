import csv,json,hashlib,datetime
from pathlib import Path
B=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
OUT=B/'hyrank_results_inventory_20261009_v1'
OUT.mkdir(exist_ok=False)
rows=[];sources={};details=[]
def read(rel):
 p=B/rel;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def metric(x):
 if isinstance(x,dict):return x['mean'],x.get('std_sample',x.get('sample_std'))
 return x,None
def add(id,name,recipe,scope,endpoint,n,budget,m,source,notes=''):
 oa,os=metric(m.get('OA',m.get('oa_percent',m.get('oa'))));aa,aas=metric(m.get('AA',m.get('aa_percent',m.get('aa'))));k,ks=metric(m.get('Kappa',m.get('kappa')))
 rows.append(dict(id=id,name=name,recipe=recipe,scope=scope,endpoint=endpoint,replicates=n,train_epochs=budget,OA_mean=oa,OA_sample_std=os,AA_mean=aa,AA_sample_std=aas,Kappa_mean=k,Kappa_sample_std=ks,source=str(B/source),notes=notes))
# Original BiDA and isolated MMD repair (not LMMD).
src='bida_hyrank_paper_repro_v4/summary.json';d=read(src)
for r in d['aggregate']:
 arm=r['arm'];add('BiDA_'+arm,'BiDA release' if arm=='released' else 'BiDA MMD argument repair','Full BiDA; native BN; no Eq9; ordinary MMD+distillation+EMA/ARS',r['protocol'],r['endpoint'],3,200,r,src,'MMD repair changes the erroneous self/self argument only; this is NOT LMMD')
details += [dict(study='BiDA_'+r['arm'],source=str(B/src),**r) for r in d['rows']]
# Eq9 full BiDA / frozen self C / full BiDA+C.
src='hyrank_ours_eq9_two_tracks_v1/summary.json';d=read(src)
meta={'B1_original':('B1 official BiDA + Eq9','Full BiDA original tokenizer; native BN; ordinary MMD+distillation+EMA/ARS+Eq9'), 'frozen_ours':('C CE+Eq9','C self Transformer+Full Joint BN; source CE+Eq9; NO LMMD'), 'bida_full_C':('B1+C','Full BiDA+C tokenizer; native BN; ordinary MMD+distillation+EMA/ARS+Eq9; NOT frozen Ours')}
for r in d['aggregate']:
 name,recipe=meta[r['group']];add(r['group'],name,recipe,r['scope'],r['endpoint'],3,200,r,src)
# Both actual A/C + CE LMMD Eq9. Ignore duplicate historical comparators.
src='hyrank_joint_lmmd_eq9_v1/summary.json';d=read(src)
for r in d['aggregate']:
 if r['group'] not in ('A','C'):continue
 arm=r['group'];add('joint_LMMD_Eq9_'+arm,arm+' CE+LMMD+Eq9',arm+' self Transformer+Full Joint BN; CE+LMMD+Eq9; no coupled/distillation/EMA',r['scope'],r['endpoint'],3,200,r,src,'Unified recipe; no source-val loader calls; unused loss initialization retained')
# Formal adapters (R0 deduplicates historical C LMMD).
src='hyrank_adapter_lmmd_formal_v1/summary.json';d=read(src)
for r in d['aggregate']:
 arm=r['arm'];names={'R0':'C LMMD no adapter (historical recipe)','R1':'C LMMD shared adapter16','R2':'C LMMD domain adapters16'}
 add('formal_adapter_'+arm,names[arm],'C self Transformer+Full Joint BN; CE+LMMD; NO Eq9; '+arm,r['scope'],r['endpoint'],3,200,r,src,'Historical source-val epoch1/every10 retained. R1 extra6144 vs R2 extra12288; same active path hidden16. R0 training reused')
# Single-factor matched seed2100 CE-only and LMMD noEq9. Latest file contains both.
src='hyrank_ce_eq9_off_2100_v1/summary.json';d=read(src)
for r in d['endpoints']:
 lm=r['group']=='CE_LMMD_Eq9_off';add('matched_2100_'+('LMMD' if lm else 'CE'),'C '+('CE+LMMD' if lm else 'CE only')+' no Eq9 (matched2100)','C self Transformer+Full Joint BN; '+('CE+LMMD' if lm else 'source CE ONLY')+'; NO Eq9; NO adapter',r['scope'],r['endpoint'],1,200,r,src,'seed2100 ONLY; best epoch%d; same initialization/data/RNG; no source-val calls'%r['epoch'])
 details.append(dict(study='matched_2100_'+('LMMD' if lm else 'CE'),source=str(B/src),seed=2100,**r))
# Original protocol fixed-only cohorts, both ordinary BN BiDA and JointBN self.
src='official_bench_v1/summary.json';d=read(src)
meta={'A':('Early official BiDA (with source-val calls)','Full BiDA ordinary BN; complete native losses; no Eq9'), 'B':('Early self A CE noEq9','Original tokenizer+Full Joint BN+self Transformer+source CE; no LMMD/Eq9'),'C':('Early self C CE noEq9','Center-relative tokenizer+Full Joint BN+self Transformer+source CE; no LMMD/Eq9')}
for arm,m in d['aggregate'].items():
 name,recipe=meta[arm];add('early_official_'+arm,name,recipe,'native','fixed200',3,200,m,src,'Historical source-val epoch1/every10; target-best not present in this summary; no recomputation done')
details += [dict(study='early_official_'+r['arm'],source=str(B/src),**r) for r in d['rows']]
# A LMMD fixed endpoint only (C deduplicated by formal R0).
src='cross_benchmark_lmmd_v1/hyrank/summary.json';d=read(src)
for r in d['aggregate']:
 if r['arm']=='A' and r['objective']=='LMMD':add('early_A_LMMD','A CE+LMMD noEq9','Original tokenizer+Full Joint BN+self Transformer+CE+LMMD; no Eq9',r['eval_protocol'],'fixed200',3,200,r,src,'Historical source-val schedule; target-best not in summary')
# Old adapters100 with Eq9; shared32/domain16 differ in active width.
src='hyrank_domain_adapter_v1/target_best_diagnostic_first100.json';d=read(src)
for r in d['results']:
 for ep,key in [('target_best','best'),('fixed100','fixed100')]:
  add('old_adapter_'+r['arm'],'Early '+r['arm']+' adapter study','C+Full Joint BN+self Transformer+CE+LMMD+Eq9; '+r['arm'],r['scope'],ep,1,100,r[key],src,'seed2100; R1 shared hidden32 vs R2 each hidden16; total extra params equal12288; differs from formal study')
  details.append(dict(study='old_adapter_'+r['arm'],source=str(B/src),seed=2100,scope=r['scope'],endpoint=ep,epoch=r['best_epoch'] if ep=='target_best' else 100,**r[key]))
# Read actual per-seed endpoint CSVs for main studies.
for folder in ('hyrank_ours_eq9_two_tracks_v1','hyrank_joint_lmmd_eq9_v1','hyrank_adapter_lmmd_formal_v1'):
 p=B/folder/'endpoints.csv';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 details += [dict(study=folder,source=str(p),**r) for r in csv.DictReader(p.open())]
# Peripheral legacy protocols: keep isolated.
src='tstnet_hyrank_native_v1/summary.json';d=read(src)
m={key:{'mean':v,'sample_std':d['sample_std'][key]} for key,v in d['means'].items()}
add('TSTnet_native','Official TSTnet native10trials','TSTnet native; source5% x10; patch12; batch100; SGD .001; native test augmentation; different protocol','native_10317_augmented','target_best_native_grid_1_11_to491',len(d['trials']),500,m,src,'10 sequential trials seed1233 once; NOT comparable to current 3seed200 grid10')
src='hyrank_clean_audit_v1/summary.json';d=read(src)
for r in d['clean_results']:
 add('legacy500_'+r['name'],'Legacy '+r['name']+' pipeline500', 'Old TSTnet-derived self pipeline, separate protocol','clean_10317','fixed500',1,500,r['metrics'],src,'seed1233; geometry/anchor bug audit; diagnostic only, not current method evidence')
# Inventory every relevant directory; read-only diagnostics and incomplete jobs are not fake metric rows.
inventory=[]
for p in sorted(B.iterdir()):
 if not p.is_dir() or p==OUT:continue
 if 'hyrank' not in p.name and p.name not in ('official_bench_v1','cross_benchmark_frozen_v1','cross_benchmark_lmmd_v1','tstnet_pipeline_bida_v1','tstnet_pipeline_self_v1'):continue
 relevant=[str(f) for f in p.glob('*summary*.json')]+[str(f) for f in p.glob('*AUDIT*')]+[str(f) for f in p.glob('*protocol*.json')]
 status=None
 f=p/'launcher_status.json'
 if f.exists():status=json.loads(f.read_text())
 inventory.append({'directory':str(p),'summary_and_audit_files':relevant,'launcher_status':status,'classification':'metric_rows_in_catalog' if any(str(p) in r['source'] for r in rows) else 'audit_diagnostic_or_duplicate_or_incomplete; inspect original files; not a new metric cohort'})
# Freeze primary reference as separate provenance, not an experimental row.
paper={'BiDA_HyRANK_OA':68.89,'Kappa':.6270,'source':'https://arxiv.org/html/2507.02268v1','table':'XIV','status':'paper reported, not our reproduction; protocol/CA aggregation caveat documented in reproduction audit'}
result={'created_local':'2026-10-09','no_training_or_model_changes':True,'groups':rows,'per_seed_details':details,'inventory':inventory,'paper_reported_separate':paper,'source_sha256':sources,'notes':['Target-best means per-seed OA maximum over that study evaluation grid; AA/Kappa use same checkpoint. It is target-label-dependent selection, separate from fixed endpoint.','Do not combine single-seed values with 3seed averages.','native10287 and full/clean10317 are separate. TSTnet legacy testing uses augmentation.','Formal adapters and matched seed2100 differ in source-val loader calls and unused-loss RNG schedule; no increment addition across protocols.','Legacy global collapse fields use varying definitions; compare zero recall and >=95% one-class collapse explicitly.']}
(OUT/'catalog.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (OUT/'all_groups.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
flat=[]
for r in details:
 v={k:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else x for k,x in r.items()};flat.append(v)
keys=sorted(set().union(*(r.keys() for r in flat)))
with (OUT/'all_seed_endpoints.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
(OUT/'source_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
def fm(m,s):return '-' if m is None else ('%.2f'%m if s is None else '%.2f +/- %.2f'%(m,s))
lines=['HyRANK results inventory (read-only consolidation)','Paper reported BiDA: OA68.89%, Kappa0.6270 (separate)','Native / full / legacy protocols remain separate. Target-best uses target GT for selection.','']
for r in rows:
 if r['scope']!='native':continue
 lines.append('%s | n=%d | epochs=%d | %s | OA %s | AA %s | Kappa %s'%(r['name'],r['replicates'],r['train_epochs'],r['endpoint'],fm(r['OA_mean'],r['OA_sample_std']),fm(r['AA_mean'],r['AA_sample_std']),fm(r['Kappa_mean'],r['Kappa_sample_std'])))
lines.extend(['','Inventory:']+[x['directory']+' | '+x['classification'] for x in inventory])
(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
# Source hashes unchanged after collection.
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
print('SAVED',OUT,'group_rows',len(rows),'seed_details',len(details),'inventory_dirs',len(inventory));print('\n'.join(lines[:24]))
