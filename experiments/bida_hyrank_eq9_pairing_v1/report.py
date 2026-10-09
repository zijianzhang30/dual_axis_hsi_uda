"""Posthoc fixed endpoint report. Never used by the sampler or training loss."""
import csv,json
import numpy as np
import study as s

def report():
 c,ns=s.environment();rows=[];paired=[];diagnostics=[];perclass=[]
 for seed in s.SEEDS:
  b0=json.loads((s.BASE/('seed_%d/frozen_manifest.json'%seed)).read_text())['evaluations']
  full0=json.loads((s.PRIOR/('baseline_%d/full_evaluation.json'%seed)).read_text())['records']
  folder=s.OUT/('B1_%d'%seed);path=folder/'frozen_manifest.json'
  if not path.exists():continue
  manifest=json.loads(path.read_text());b1=manifest['evaluations']
  # Target GT enters here, after training and all selections are frozen, for diagnosis only.
  args=c.options(seed);values,_=ns['loaders'](args);ds=values[-1].dataset
  gt=ds.labels.astype(np.int64)
  for epoch in range(1,201):
   p=np.load(folder/'pairing'/('epoch_%03d.npz'%epoch));matched=p['matched'];truth=gt[p['selected']]
   cm=np.zeros((12,12),np.int64);np.add.at(cm,(p['source'],truth),1)
   diag={'seed':seed,'epoch':epoch,'matched_coverage':float(matched.mean()),
    'matched_true_class_agreement':float((truth[matched]==p['source'][matched]).mean()) if matched.any() else None,
    'all_rows_true_class_agreement':float((truth==p['source']).mean()),
    'unmatched_source_class_counts':np.bincount(p['source'][~matched],minlength=12).tolist(),
    'source_true_vs_selected_target_true_confusion':cm.tolist()}
   diagnostics.append(diag)
  for scope in ('native','paper_full'):
   key='metrics' if scope=='native' else 'paper_full_metrics'
   base=b0 if scope=='native' else full0
   for endpoint in ('fixed200','target_best_diagnostic'):
    r0=base[-1] if endpoint=='fixed200' else max(base,key=lambda r:r[key]['oa_percent'])
    r1=b1[-1] if endpoint=='fixed200' else max(b1,key=lambda r:r[key]['oa_percent'])
    m0=r0[key];m1=r1[key]
    for group,r,m in [('B0',r0,m0),('B1',r1,m1)]:
     recall=m['per_class_percent'];zero=[i+1 for i,v in enumerate(recall) if v==0]
     row={'seed':seed,'scope':scope,'endpoint':endpoint,'group':group,'epoch':r['epoch'],
      'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],'zero_recall_classes':zero}
     rows.append(row)
     for k,v in enumerate(recall):perclass.append(dict(seed=seed,scope=scope,endpoint=endpoint,group=group,class_id=k+1,recall=v))
    paired.append({'seed':seed,'scope':scope,'endpoint':endpoint,'B1_epoch':r1['epoch'],'B0_epoch':r0['epoch'],
     'OA_delta_pp':m1['oa_percent']-m0['oa_percent'],'AA_delta_pp':m1['aa_percent']-m0['aa_percent'],'Kappa_delta':m1['kappa']-m0['kappa']})
 def csvfile(name,data):
  if not data:return
  with (s.OUT/name).open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 csvfile('endpoints.csv',rows);csvfile('paired_deltas.csv',paired);csvfile('per_class_recall.csv',perclass)
 s.dump(s.OUT/'offline_pair_correspondence.json',diagnostics)
 aggregates=[]
 for scope in ('native','paper_full'):
  for endpoint in ('fixed200','target_best_diagnostic'):
   for group in ('B0','B1'):
    selected=[r for r in rows if r['scope']==scope and r['endpoint']==endpoint and r['group']==group]
    if selected:
     aggregates.append({'scope':scope,'endpoint':endpoint,'group':group,'n':len(selected),
      **{key:{'mean':float(np.mean([r[key] for r in selected])), 'std_sample':float(np.std([r[key] for r in selected],ddof=1)) if len(selected)>1 else None} for key in ('OA','AA','Kappa')}})
 s.dump(s.OUT/'summary.json',{'completed_seeds':sorted(set(r['seed'] for r in rows)),'aggregate':aggregates,
 'paired':paired,'paper_reported_OA':68.89,'paper_reported_numbers_separate':True,
 'target_best_is_only_diagnostic':True,'no_combination_with_MMD_repair':True,'followups_launched':False})
 print(json.dumps(aggregates,indent=2))
if __name__=='__main__':report()
