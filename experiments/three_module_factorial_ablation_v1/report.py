import csv,json,subprocess,sys
from collections import defaultdict
import numpy as np
import study as s
def write(name,rows):
 with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(y,p,k):
 cm=np.zeros((k,k),dtype=np.int64);np.add.at(cm,(y,p),1);oa=np.trace(cm)/cm.sum();pe=(cm.sum(0)*cm.sum(1)).sum()/cm.sum()**2;rec=100*cm.diagonal()/cm.sum(1);return {'OA':float(100*oa),'AA':float(rec.mean()),'Kappa':float((oa-pe)/(1-pe))},rec
def main():
 s.verify()
 for d,r,seed in s.new_tasks():
  mf=json.loads((s.OUT/d/('%s_%d'%(r,seed))/'TRAINING_PREDICTIONS_FROZEN.json').read_text());assert mf['epochs']==200 and mf['all200_reference_data_and_RNG_match']
  for q in mf['checkpoints']+mf['predictions']:assert s.sha(q['path'])==q['sha256']
 for d in s.DATASETS:subprocess.run([sys.executable,str(s.CODE/'assemble.py'),'--dataset',d],check=True)
 all_records=[]
 for d in s.DATASETS:all_records+=json.loads((s.OUT/('ALL_'+d+'_PREDICTIONS_FROZEN.json')).read_text())['records']
 assert len(all_records)==960;files={q['path']:q['sha256'] for q in all_records}
 for p,h in files.items():assert s.sha(p)==h
 s.dump(s.OUT/'GLOBAL_BEFORE_GT_FREEZE.json',{'all21_new_trainings_and_all72_group_grid_predictions_frozen':True,'files':files});rows=[];perclass=[];diag=[]
 for d in s.DATASETS:
  gp=s.OUT/('GT_SCORING_ONLY_'+d+'.npz');subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,sys.argv[1]);import study as s,numpy as np;m,v,r=s.setup(sys.argv[2],"J_C",2100);np.savez_compressed(sys.argv[3],labels=v[-1].dataset.label,indices=v[-1].dataset.indices,classes=r["classes"])',str(s.CODE),d,str(gp)],check=True);a=np.load(gp);gt=a['labels'];k=int(a['classes'])
  for q in all_records:
   if q['dataset']!=d:continue
   f=np.load(q['path']);xy=f['coordinates'];y=gt[xy[:,0],xy[:,1]];assert y.min()>=0 and y.max()<k
   if q['scope']=='native':assert np.array_equal(xy,a['indices'])
   for dual,field in ((0,'relation_predictions'),(1,'fusion_predictions')):
    p=f[field];mm,rec=metrics(y,p,k);group=q['relation']+'_'+str(dual);row={'dataset':d,'scope':q['scope'],'group':group,'seed':q['seed'],'epoch':q['epoch'],**mm};rows.append(row)
    if q['epoch']==200:
     for c,v in enumerate(rec):perclass.append({'dataset':d,'scope':q['scope'],'group':group,'seed':q['seed'],'class':c+1,'count':int((y==c).sum()),'recall':float(v)})
   if q['epoch']==200:
    c0=f['relation_predictions']==y;c1=f['spectral_predictions']==y;cf=f['fusion_predictions']==y;dis=~f['agreement'];diag.append({'dataset':d,'scope':q['scope'],'relation':q['relation'],'seed':q['seed'],'recovery_rate':float(100*cf[~c0].mean()) if (~c0).any() else None,'damage_rate':float(100*(~cf[c0]).mean()) if c0.any() else None,'disagreement_Rel_accuracy':float(100*c0[dis].mean()) if dis.any() else None,'disagreement_Spec_accuracy':float(100*c1[dis].mean()) if dis.any() else None,'disagreement_Fusion_accuracy':float(100*cf[dis].mean()) if dis.any() else None,'new_zero_classes':[c+1 for c in range(k) if ((f['relation_predictions']==c)&(y==c)).any() and not ((f['fusion_predictions']==c)&(y==c)).any()]})
 write('all_grid_scores_ORACLE_DIAGNOSTIC.csv',rows);fixed=[r for r in rows if r['epoch']==200];write('fixed200_per_seed.csv',fixed);write('fixed200_per_class.csv',perclass);write('fixed200_recovery_damage_disagreement.csv',diag);gg=defaultdict(list)
 for r in fixed:gg[r['dataset'],r['scope'],r['group']].append(r)
 summary=[]
 for (d,scope,g),rs in sorted(gg.items()):summary.append({'dataset':d,'scope':scope,'group':g,**{key+suffix:float(fn([r[key] for r in rs])) for key in ('OA','AA','Kappa') for suffix,fn in (('_mean',np.mean),('_std',lambda a:np.std(a,ddof=1)))}})
 write('fixed200_mean_std.csv',summary);pairs=[]
 for d in s.DATASETS:
  for scope in (('native','full') if d=='hyrank' else ('native',)):
   for seed in s.SEEDS:
    z={r['group']:r for r in fixed if r['dataset']==d and r['scope']==scope and r['seed']==seed}
    for label,a,b in [('BN_progressive','J_A_0','N_A_0'),('C_progressive','J_C_0','J_A_0'),('Evidence_progressive','J_C_1','J_C_0'),('full_minus_BN','J_C_1','N_C_1'),('full_minus_C','J_C_1','J_A_1'),('full_minus_Evidence','J_C_1','J_C_0')]:pairs.append({'dataset':d,'scope':scope,'seed':seed,'contrast':label,**{'delta_'+key:z[a][key]-z[b][key] for key in ('OA','AA','Kappa')}})
 write('paired_component_deltas.csv',pairs);effects=[]
 for d in s.DATASETS:
  for scope in (('native','full') if d=='hyrank' else ('native',)):
   for seed in s.SEEDS:
    values={((1 if r['group'][0]=='J' else 0),(1 if r['group'][2]=='C' else 0),int(r['group'][-1])):r for r in fixed if r['dataset']==d and r['scope']==scope and r['seed']==seed}
    for key in ('OA','AA','Kappa'):
     # Difference-of-differences: no target-dependent choice or sign assumption.
     for subset in ((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)):
      total=0.
      for bits,v in values.items():total+=(-1)**(len(subset)-sum(bits[j] for j in subset))*v[key]
      total/=2**(3-len(subset));effects.append({'dataset':d,'scope':scope,'seed':seed,'metric':key,'effect':'x'.join(('BN','C','Evidence')[j] for j in subset),'value':total})
 write('factorial_effects_and_interactions_per_seed.csv',effects);best=defaultdict(list)
 for r in rows:best[r['dataset'],r['scope'],r['group'],r['seed']].append(r)
 write('same_grid_target_best_ORACLE_ONLY.csv',[max(rs,key=lambda r:r['OA']) for rs in best.values()]);text='three_module_factorial_ablation_v1\nCompleted all locked21 new trainings; existing15 relation cases and all9 independent spectral experts reused.\nGroups: N/J=Native/Joint BN; A/C=original/C tokenizer; _0/_1=without/with fixed Product-of-Evidence.\nFormal fixed200 only; mean+sample std; all960 grid files frozen and hashed before new-study target GT scoring.\nBN-off includes native momentum .1/two updates; BN-on includes established .19/one update. This ablates the established component, not the estimator separately from momentum.\nNo tuning/new variants/automatic follow-up training. Do not claim all components help all classes; inspect paired and per-class results.\n'+ '\n'.join(json.dumps(r) for r in summary)+'\n'
 (s.OUT/'REPORT.txt').write_text(text);(s.OUT/'FINDINGS.txt').write_text(text);s.verify();s.dump(s.OUT/'REPORT_COMPLETE.json',{'fixed200_formal':True,'all72_groups_scored':True,'no_followup_started':True});print('FACTORIAL_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
