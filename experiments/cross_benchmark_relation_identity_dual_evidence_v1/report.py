import csv,json,subprocess,sys
import numpy as np
import study as s

def write(name,rows):
 with (s.OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(y,p,k):
 cm=np.zeros((k,k),dtype=np.int64);np.add.at(cm,(y,p),1);n=cm.sum();oa=np.trace(cm)/n;pe=(cm.sum(0)*cm.sum(1)).sum()/n**2;rec=np.divide(cm.diagonal(),cm.sum(1),out=np.full(k,np.nan),where=cm.sum(1)>0)
 return {'OA':float(100*oa),'AA':float(100*np.nanmean(rec)),'Kappa':float((oa-pe)/(1-pe))},100*rec
def ratio(mask,hit):return 100*float(hit[mask].mean()) if mask.any() else None
def main():
 s.verify();barrier={};manifests={}
 # No target label values are loaded until ALL six runs/grid predictions verify.
 for d in s.DATASETS:
  for seed in s.SEEDS:
   f=s.OUT/('%s_%d'%(d,seed));tm=json.loads((f/'SPECTRAL_TRAINING_COMPLETE.json').read_text());assert tm['epochs']==200 and tm['all200_augmented_batches_and_CPU_CUDA_RNG_match_R0'];mf=json.loads((f/'ALL_PREDICTIONS_FROZEN.json').read_text());manifests[d,seed]=mf
   for name in ('SPECTRAL_TRAINING_COMPLETE.json','ALL_PREDICTIONS_FROZEN.json','coordinates_frozen.npz','history.jsonl'):barrier[str(f/name)]=s.sha(f/name)
   for r in tm['checkpoints']+mf['records']:assert s.sha(r['path'])==r['sha256'];barrier[r['path']]=r['sha256']
 s.dump(s.OUT/'GLOBAL_BEFORE_GT_FREEZE.json',{'all_six_200epoch_trainings_and_all_predictions_verified':True,'files':barrier})
 rows=[];classes=[];diag=[];oracle=[]
 for d in s.DATASETS:
  # Each native snapshot has its own modules named models/utils. Isolate imports
  # so Pavia's modules cannot replace HyRANK's author-out68 loader/class mapping.
  gp=s.OUT/('GT_scoring_ONLY_'+d+'.npz')
  subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,sys.argv[1]);import study as s,numpy as np;m,values,r=s.setup(sys.argv[2],2100);np.savez_compressed(sys.argv[3],labels=values[-1].dataset.label,indices=values[-1].dataset.indices,classes=r["classes"])',str(s.CODE),d,str(gp)],check=True)
  gg=np.load(gp);gt=gg['labels'];k=int(gg['classes'])
  for seed in s.SEEDS:
   for r in manifests[d,seed]['records']:
    a=np.load(r['path']);xy=a['coordinates'];y=gt[xy[:,0],xy[:,1]].astype(int);assert y.min()>=0 and y.max()<k
    if r['scope']=='native':assert np.array_equal(xy,gg['indices'])
    for name in ('R0','R1','R3'):
     p=a[name+'_predictions'];mm,rec=metric(y,p,k);entry={'benchmark':d,'scope':r['scope'],'seed':seed,'epoch':r['epoch'],'model':name,**mm};oracle.append(entry)
     if r['epoch']==200:
      rows.append(entry)
      for c,v in enumerate(rec):classes.append({'benchmark':d,'scope':r['scope'],'seed':seed,'model':name,'class':c+1,'count':int((y==c).sum()),'recall':v})
    if r['epoch']==200:
     c0=a['R0_predictions']==y;c1=a['R1_predictions']==y;c3=a['R3_predictions']==y;dis=a['disagreement'];both=~c0&~c1
     diag.append({'benchmark':d,'scope':r['scope'],'seed':seed,'R0_error_count':int((~c0).sum()),'R3_recovered_count':int((~c0&c3).sum()),'R3_recovery_rate':ratio(~c0,c3),'R0_correct_count':int(c0.sum()),'R3_damaged_count':int((c0&~c3).sum()),'R3_damage_rate':ratio(c0,~c3),'disagreement_count':int(dis.sum()),'disagreement_rate':100*dis.mean(),'disagreement_R0_accuracy':ratio(dis,c0),'disagreement_R1_accuracy':ratio(dis,c1),'disagreement_R3_accuracy':ratio(dis,c3),'both_top1_wrong_count':int(both.sum()),'both_wrong_R3_recovery_rate':ratio(both,c3),'R3_max_prediction_share':100*np.bincount(a['R3_predictions'],minlength=k).max()/len(y)})
 write('fixed200_per_seed.csv',rows);write('fixed200_per_class.csv',classes);write('recovery_damage_disagreement.csv',diag);write('same_grid_oracle_diagnostic.csv',oracle)
 summary=[];deltas=[]
 for d in s.DATASETS:
  for scope in ('native','full'):
   subset=[r for r in rows if r['benchmark']==d and r['scope']==scope]
   if not subset:continue
   for name in ('R0','R1','R3'):
    rs=[r for r in subset if r['model']==name];summary.append({'benchmark':d,'scope':scope,'model':name,**{t+suffix:float(fun([r[t] for r in rs])) for t in ('OA','AA','Kappa') for suffix,fun in (('_mean',np.mean),('_std',lambda a:np.std(a,ddof=1)))}})
   for seed in s.SEEDS:
    v={r['model']:r for r in subset if r['seed']==seed};deltas.append({'benchmark':d,'scope':scope,'seed':seed,**{'delta_'+t:v['R3'][t]-v['R0'][t] for t in ('OA','AA','Kappa')}})
 write('fixed200_summary.csv',summary);write('same_seed_deltas.csv',deltas)
 checks={}
 for d in s.DATASETS:
  rs=[r for r in deltas if r['benchmark']==d and r['scope']=='native'];means={t:np.mean([r['delta_'+t] for r in rs]) for t in ('OA','AA')};neg={t:sum(r['delta_'+t]<0 for r in rs) for t in means};checks[d]={'mean_delta':means,'positive_seeds':{t:3-neg[t] for t in means},'both_means_improve':all(v>0 for v in means.values()),'systematic_regression':any(means[t]<-.5 and neg[t]>=2 for t in means)}
 go=any(v['both_means_improve'] for v in checks.values()) and not any(v['systematic_regression'] for v in checks.values());verdict='Go' if go else 'No-Go / Houston-specific';s.dump(s.OUT/'decision.json',{'verdict':verdict,'checks':checks,'no_followup_started':True})
 lines=['cross_benchmark_relation_identity_dual_evidence_v1',verdict,'Main: fixed200, native protocol, sample standard deviation ddof=1.','R0 immutable formal C reused; only independent source-CE spectral expert trained.','All six runs and all grid predictions/hash verified before target GT scoring.','No tuning, checkpoint selection or follow-up training.','']
 for r in summary:lines.append('%s %s %s: OA %.6f +/- %.6f; AA %.6f +/- %.6f; Kappa %.7f +/- %.7f'%(r['benchmark'],r['scope'],r['model'],r['OA_mean'],r['OA_std'],r['AA_mean'],r['AA_std'],r['Kappa_mean'],r['Kappa_std']))
 lines+=['','Same-seed deltas:']+[json.dumps(r) for r in deltas]+['','Recovery/damage/disagreement:']+[json.dumps(r) for r in diag]+['','Go checks: '+json.dumps(checks),'HyRANK full-10317 is secondary; native-10287 is the formal primary protocol.','Target-best is oracle-only, see same_grid_oracle_diagnostic.csv; never used in main results.','Per-class harm and zero-recall must be inspected in fixed200_per_class.csv.']
 text='\n'.join(lines)+'\n';(s.OUT/'REPORT.txt').write_text(text);(s.OUT/'FINDINGS.txt').write_text(text);print(text)
if __name__=='__main__':main()
