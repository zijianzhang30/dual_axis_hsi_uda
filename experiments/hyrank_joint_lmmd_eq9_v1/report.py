"""Target GT first scored here AFTER all six 200-epoch runs are frozen."""
import csv,json
import numpy as np
import study as s

def main():
 assert all((s.OUT/('%s_%d/frozen_manifest.json'%(a,seed))).exists() for a in s.ARMS for seed in s.SEEDS)
 c,ns=s.environment();args=c.options(2100);loaders,data=ns['loaders'](args);ds=loaders[-1].dataset
 native_labels=np.asarray(ds.labels,dtype=np.int64);geom=s.geometry();_,_,_,_,full_labels=geom.full_geometry(ds)
 rows=[];class_rows=[];pairs=[];groups={};timings=[]
 def add(group,seed,ev):
  groups[group,seed]=ev
  for scope,key in [('native','metrics'),('full','paper_full_metrics')]:
   for endpoint in ('fixed200','target_best_diagnostic'):
    r=ev[-1] if endpoint=='fixed200' else max(ev,key=lambda r:r[key]['oa_percent']);m=r[key]
    rows.append({'group':group,'seed':seed,'scope':scope,'endpoint':endpoint,'epoch':r['epoch'],
     'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],
     'max_prediction_share':max(m['prediction_shares']),'single_class_collapse_at95pct':max(m['prediction_shares'])>=.95,
     'zero_recall_classes':[k+1 for k,v in enumerate(m['per_class_percent']) if v==0]})
    for k,v in enumerate(m['per_class_percent']):class_rows.append({'group':group,'seed':seed,'scope':scope,'endpoint':endpoint,'class':k+1,'recall':v})
 for a in s.ARMS:
  for seed in s.SEEDS:
   folder=s.OUT/('%s_%d'%(a,seed));manifest=json.loads((folder/'frozen_manifest.json').read_text());ev=manifest['evaluations']
   for r in ev:
    for scope,key,labels in [('native','metrics',native_labels),('full','paper_full_metrics',full_labels)]:
     z=np.load(r[scope+'_prediction_path']);r[key]=c.helper.metrics(labels,z['predictions'],z['logits'],12)
   s.dump(folder/'offline_scored_evaluations.json',ev);add(a,seed,ev)
   timings.append({'arm':a,'seed':seed,'parameters':manifest['config']['parameters'],'train_seconds':manifest['train_seconds']})
   # Matching A/C management and backbone, actual prediction-dependent target ids need not match.
   if a=='C':
    ac=json.loads((s.OUT/('A_%d/config.json'%seed)).read_text());assert ac['shared_initial_hash']==manifest['config']['shared_initial_hash']
    h0=[json.loads(x) for x in (s.OUT/('A_%d/history.jsonl'%seed)).read_text().splitlines()]
    h1=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
    for x,y in zip(h0,h1):
     for key in ('batch_stream_sha256','actual_centers_sha256','rng_stream_sha256','all_rng_stream_sha256'):assert x[key]==y[key]
 historical=json.loads((s.HIST/'hyrank/summary.json').read_text())['rows']
 # Fixed200-only legacy CE+LMMD, separately save historical comparisons; do not invent missing best trajectories.
 for seed in s.SEEDS:
  add('C_CE_Eq9_historical',seed,json.loads((s.PRIOR/('frozen_ours_%d/frozen_manifest.json'%seed)).read_text())['evaluations'])
  add('B1_C_different_arch_loss',seed,json.loads((s.PRIOR/('bida_full_C_%d/frozen_manifest.json'%seed)).read_text())['evaluations'])
 for scope,key in [('native','metrics'),('full','paper_full_metrics')]:
  for seed in s.SEEDS:
   for endpoint in ('fixed200','target_best_diagnostic'):
    def select(group):
     ev=groups[group,seed];return ev[-1][key] if endpoint=='fixed200' else max(ev,key=lambda r:r[key]['oa_percent'])[key]
    for comp in ('A','C_CE_Eq9_historical','B1_C_different_arch_loss'):
     a=select('C');b=select(comp)
     pairs.append({'seed':seed,'scope':scope,'endpoint':endpoint,'comparison':'C minus '+comp,
      'OA_delta_pp':a['oa_percent']-b['oa_percent'],'AA_delta_pp':a['aa_percent']-b['aa_percent'],'Kappa_delta':a['kappa']-b['kappa']})
   if scope=='native':
    old=next(r for r in historical if r['eval_protocol']=='native' and r['arm']=='C' and r['objective']=='LMMD' and r['seed']==seed)
    a=select('C') if False else groups['C',seed][-1]['metrics']
    pairs.append({'seed':seed,'scope':scope,'endpoint':'fixed200','comparison':'C minus historical_C_CE_LMMD_noEq9_DIFFERENT_source_val_schedule',
      'OA_delta_pp':a['oa_percent']-old['oa_percent'],'AA_delta_pp':a['aa_percent']-old['aa_percent'],'Kappa_delta':a['kappa']-old['kappa']})
 aggregates=[]
 for group in ('A','C','C_CE_Eq9_historical','B1_C_different_arch_loss'):
  for scope in ('native','full'):
   for endpoint in ('fixed200','target_best_diagnostic'):
    v=[r for r in rows if r['group']==group and r['scope']==scope and r['endpoint']==endpoint]
    aggregates.append({'group':group,'scope':scope,'endpoint':endpoint,**{k:{'mean':float(np.mean([r[k] for r in v])),'std_sample':float(np.std([r[k] for r in v],ddof=1))} for k in ('OA','AA','Kappa')}})
 for name,v in [('endpoints.csv',rows),('per_class_recall.csv',class_rows),('paired_deltas.csv',pairs),('compute.csv',timings)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
 s.dump(s.OUT/'summary.json',{'aggregate':aggregates,'paired':pairs,'formal_endpoint':'fixed200','A_C_streams_and_initial_backbone_verified':True,
 'target_best_diagnostic_only':True,'historical_noEq9_has_different_source_val_iteration_schedule':True,
 'B1_C_has_different_BN_structure_and_complete_adaptation_losses':True,'training_count':6,'automatic_followups':False})
 print(json.dumps(aggregates,indent=2),flush=True)
if __name__=='__main__':main()
