import json,csv,numpy as np
import study as s

def main():
 c=s.environment();model,loaders,helper,classes,audit=c.setup('hyrank','C',2100);ds=loaders[-1].dataset;clean=c.clean_dataset(ds,keep_labels=True)
 labels={'native':np.asarray(ds.labels,dtype=np.int64),'full':np.asarray(clean.labels,dtype=np.int64)}
 rows=[];class_rows=[];paired=[];groups={}
 for arm in ('R0',)+s.ARMS:
  for seed in s.SEEDS:
   f=s.OUT/('%s_%d'%(arm,seed));p=json.loads((f/'offline_predictions_frozen.json').read_text());ev=p['evaluations'];groups[arm,seed]=ev
   for scope,key in [('native','metrics'),('full','paper_full_metrics')]:
    for r in ev:
     z=np.load(r[scope+'_prediction_path']);r[key]=helper.metrics(labels[scope],z['predictions'],z['logits'],12)
    for endpoint in ('fixed200','target_best'):
     r=ev[-1] if endpoint=='fixed200' else max(ev,key=lambda r:r[key]['oa_percent']);m=r[key]
     rows.append({'arm':arm,'seed':seed,'scope':scope,'endpoint':endpoint,'epoch':r['epoch'],'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],
     'single_class_collapse95':max(m['prediction_shares'])>=.95,'zero_recall_classes':[k+1 for k,v in enumerate(m['per_class_percent']) if v==0]})
     for k,v in enumerate(m['per_class_percent']):class_rows.append({'arm':arm,'seed':seed,'scope':scope,'endpoint':endpoint,'class':k+1,'recall':v})
   s.dump(f/'offline_scored_evaluations.json',ev)
 for scope,key in [('native','metrics'),('full','paper_full_metrics')]:
  for endpoint in ('fixed200','target_best'):
   for seed in s.SEEDS:
    def selected(arm):
     ev=groups[arm,seed];return ev[-1][key] if endpoint=='fixed200' else max(ev,key=lambda r:r[key]['oa_percent'])[key]
    for b in ('R0','R1'):
     v=selected('R2');w=selected(b);paired.append({'scope':scope,'endpoint':endpoint,'seed':seed,'comparison':'R2 minus '+b,
     'OA_delta_pp':v['oa_percent']-w['oa_percent'],'AA_delta_pp':v['aa_percent']-w['aa_percent'],'Kappa_delta':v['kappa']-w['kappa']})
 aggregates=[]
 for scope in ('native','full'):
  for endpoint in ('fixed200','target_best'):
   for arm in ('R0',)+s.ARMS:
    v=[r for r in rows if r['scope']==scope and r['endpoint']==endpoint and r['arm']==arm]
    aggregates.append({'scope':scope,'endpoint':endpoint,'arm':arm,**{k:{'mean':float(np.mean([r[k] for r in v])),'std_sample':float(np.std([r[k] for r in v],ddof=1))} for k in ('OA','AA','Kappa')}})
 for name,v in [('endpoints.csv',rows),('paired_deltas.csv',paired),('per_class_recall.csv',class_rows)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
 s.dump(s.OUT/'summary.json',{'aggregate':aggregates,'paired':paired,'R0_training_reused':True,'new_training_count':6,
 'R1_R2_same_path_width16_different_total_params':True,'target_best_grid10_to200_only':True,'no_auto_followups':True})
 print(json.dumps(aggregates,indent=2),flush=True)
if __name__=='__main__':main()
