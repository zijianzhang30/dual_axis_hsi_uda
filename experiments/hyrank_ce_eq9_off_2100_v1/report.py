import json,csv,numpy as np
import study as s

def main():
 c,ns=s.environment();args=c.options(2100);ds=ns['loaders'](args)[0][-1].dataset
 labels={'native':np.asarray(ds.labels,dtype=np.int64),'full':s.geometry().full_geometry(ds)[-1]}
 folder=s.OUT/'C_2100';manifest=json.loads((folder/'frozen_manifest.json').read_text());assert manifest['completed_epochs']==200
 ev=manifest['evaluations'];control=json.loads((s.CONTROL/'C_2100/offline_scored_evaluations.json').read_text())
 trajectory=[];endpoints=[];perclass=[]
 for scope,key in [('native','metrics'),('full','paper_full_metrics')]:
  for r in ev:
   z=np.load(r[scope+'_prediction_path']);r[key]=c.helper.metrics(labels[scope],z['predictions'],z['logits'],12)
   b=next(b[key] for b in control if b['epoch']==r['epoch']);m=r[key]
   trajectory.append({'scope':scope,'epoch':r['epoch'],'CE_OA':m['oa_percent'],'CE_LMMD_OA':b['oa_percent'],'OA_delta':m['oa_percent']-b['oa_percent'],
    'CE_AA':m['aa_percent'],'CE_LMMD_AA':b['aa_percent'],'CE_C7':m['per_class_percent'][6],'CE_LMMD_C7':b['per_class_percent'][6],
    'CE_C8':m['per_class_percent'][7],'CE_LMMD_C8':b['per_class_percent'][7]})
  for endpoint in ('fixed200','target_best'):
   off=ev[-1] if endpoint=='fixed200' else max(ev,key=lambda r:r[key]['oa_percent']);on=control[-1] if endpoint=='fixed200' else max(control,key=lambda r:r[key]['oa_percent'])
   for group,r in [('CE_only_Eq9_off',off),('CE_LMMD_Eq9_off',on)]:
    m=r[key];endpoints.append({'scope':scope,'endpoint':endpoint,'group':group,'epoch':r['epoch'],
     'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],'zero_recall_classes':[k+1 for k,v in enumerate(m['per_class_percent']) if v==0],
     'single_class_collapse_at95pct':max(m['prediction_shares'])>=.95})
    for k,v in enumerate(m['per_class_percent']):perclass.append({'scope':scope,'endpoint':endpoint,'group':group,'class':k+1,'recall':v})
 for name,rows in [('trajectory.csv',trajectory),('endpoints.csv',endpoints),('per_class_recall.csv',perclass)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 s.dump(folder/'offline_scored_evaluations.json',ev);s.dump(s.OUT/'summary.json',{'seed':2100,'endpoints':endpoints,'trajectory':trajectory,'single_factor_only':True,'single_factor':'LMMD removed; Eq9 disabled in BOTH groups',
 'no_other_variants_launched':True,'target_best_separate_from_fixed200':True,'target_GT_scored_after_freezing_all200_outputs':True})
 print(json.dumps(endpoints,indent=2),flush=True)
if __name__=='__main__':main()
