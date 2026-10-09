import csv,json
import numpy as np
import study as s

def main():
 rows=[];deltas=[];classes=[];aggregates=[];timings=[]
 def collapse(m):
  shares=m['prediction_shares'];return {'zero_recall_classes':[i+1 for i,v in enumerate(m['per_class_percent']) if v==0],
   'max_prediction_share':max(shares),'dominant_prediction_class':int(np.argmax(shares))+1,
   'near_single_class_collapse_at_95pct':max(shares)>=.95}
 for seed in s.SEEDS:
  baseline=json.loads((s.B1/('B1_%d/frozen_manifest.json'%seed)).read_text())['evaluations']
  groups={'B1_original':baseline}
  for track in s.TRACKS:
   manifest=json.loads((s.OUT/('%s_%d/frozen_manifest.json'%(track,seed))).read_text());groups[track]=manifest['evaluations']
   timings.append({'seed':seed,'track':track,'parameters':manifest['config']['parameters'],'train_seconds':manifest['train_seconds']})
  for scope,key in [('native','metrics'),('full','paper_full_metrics')]:
   for endpoint in ('fixed200','OA_best'):
    selected={group:(recs[-1] if endpoint=='fixed200' else max(recs,key=lambda r:r[key]['oa_percent'])) for group,recs in groups.items()}
    for group,r in selected.items():
     m=r[key];rows.append({'seed':seed,'scope':scope,'endpoint':endpoint,'group':group,'selected_epoch':r['epoch'],
     'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],**collapse(m)})
     for k,v in enumerate(m['per_class_percent']):classes.append({'seed':seed,'scope':scope,'endpoint':endpoint,'group':group,'class_id':k+1,'recall':v})
     if group!='B1_original':
      b=selected['B1_original'][key]
      deltas.append({'seed':seed,'scope':scope,'endpoint':endpoint,'comparison':group+' minus B1_original',
      'OA_delta_pp':m['oa_percent']-b['oa_percent'],'AA_delta_pp':m['aa_percent']-b['aa_percent'],'Kappa_delta':m['kappa']-b['kappa']})
 for scope in ('native','full'):
  for endpoint in ('fixed200','OA_best'):
   for group in ('B1_original',)+s.TRACKS:
    a=[r for r in rows if r['scope']==scope and r['endpoint']==endpoint and r['group']==group]
    aggregates.append({'scope':scope,'endpoint':endpoint,'group':group,
    **{k:{'mean':float(np.mean([r[k] for r in a])),'std_sample':float(np.std([r[k] for r in a],ddof=1))} for k in ('OA','AA','Kappa')}})
 for name,values in [('endpoints.csv',rows),('paired_deltas.csv',deltas),('per_class_recall.csv',classes),('compute.csv',timings)]:
  with (s.OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 s.dump(s.OUT/'summary.json',{'aggregate':aggregates,'paired':deltas,'compute':timings,'paper_reported_BiDA_OA_separate':68.89,
 'best_endpoint':'target-selected highest OA every10 epochs; AA/Kappa taken at same epoch; not target-independent selection',
 'frozen_ours_comparison_has_objective_and_BN_differences':True,'bida_full_C_comparison_changes_only_tokenizer':True,'no_followups':True})
 print(json.dumps(aggregates,indent=2),flush=True)
if __name__=='__main__':main()
