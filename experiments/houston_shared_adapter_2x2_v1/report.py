import csv,json,numpy as np
import study as s

def main():
 c=s.environment();m,loaders,helper,classes,audit=c.setup('houston','C',2100);labels=np.asarray(loaders[-1].dataset.labels);idx=loaders[-1].dataset.indices
 endpoints=[];perclass=[];compute=[]
 for group in s.GROUPS:
  for seed in s.SEEDS:
   folder=s.OUT/('%s_%d'%(group,seed));assert np.array_equal(np.load(folder/'evaluation_indices.npy'),idx)
   frozen=json.loads((folder/'offline_predictions_frozen.json').read_text());assert [r['epoch'] for r in frozen['records']]==list(range(10,201,10));records=[]
   for r in frozen['records']:
    assert s.sha(r['path'])==r['sha256'];z=np.load(r['path']);records.append({'epoch':r['epoch'],'metrics':helper.metrics(labels,z['predictions'],z['logits'],7)})
   s.dump(folder/'offline_scored_evaluations.json',records)
   for endpoint,r in [('fixed200',records[-1]),('target_best',max(records,key=lambda r:r['metrics']['oa_percent']))]:
    mm=r['metrics'];endpoints.append({'group':group,'seed':seed,'endpoint':endpoint,'epoch':r['epoch'],'OA':mm['oa_percent'],'AA':mm['aa_percent'],'Kappa':mm['kappa'],'zero_recall_classes':[k+1 for k,v in enumerate(mm['per_class_percent']) if v==0],'single_class_collapse95':max(mm['prediction_shares'])>=.95})
    perclass.extend([{'group':group,'seed':seed,'endpoint':endpoint,'class':k+1,'recall':v} for k,v in enumerate(mm['per_class_percent'])])
   origin=folder if group.endswith('1') else s.baseline(group,seed);mf=json.loads((origin/'frozen_manifest.json').read_text());cfg=mf.get('config',json.loads((origin/'config.json').read_text()));compute.append({'group':group,'seed':seed,'parameters':cfg['parameters'],'train_seconds':mf['train_seconds'],'source_val':mf.get('source_val_at_final')})
 agg=[];paired=[];interactions=[]
 for ep in ('fixed200','target_best'):
  for group in s.GROUPS:
   rs=[r for r in endpoints if r['endpoint']==ep and r['group']==group];agg.append({'endpoint':ep,'group':group,**{k:{'mean':float(np.mean([r[k] for r in rs])),'std_sample':float(np.std([r[k] for r in rs],ddof=1))} for k in ('OA','AA','Kappa')}})
  for seed in s.SEEDS:
   r={x['group']:x for x in endpoints if x['endpoint']==ep and x['seed']==seed}
   for aa,bb in [('CE1','CE0'),('LMMD1','LMMD0'),('LMMD1','CE0')]:paired.append({'endpoint':ep,'seed':seed,'comparison':aa+'-'+bb,**{k+'_delta':r[aa][k]-r[bb][k] for k in ('OA','AA','Kappa')}})
   interactions.append({'endpoint':ep,'seed':seed,'definition':'(LMMD1-LMMD0)-(CE1-CE0)',**{k+'_interaction':(r['LMMD1'][k]-r['LMMD0'][k])-(r['CE1'][k]-r['CE0'][k]) for k in ('OA','AA','Kappa')}})
 s.dump(s.OUT/'summary.json',{'aggregate':agg,'paired':paired,'interaction':interactions,'compute':compute,'HyRANK_reused_summary':str(s.OUT/'HyRANK_reused_summary.json'),'new_training_count':6,'Pavia_deferred':True,'candidate_screening_not_independent_generalization':True,'target_best_selection_uses_GT_separate_from_fixed200':True})
 for name,rs in [('endpoints.csv',endpoints),('paired_deltas.csv',paired),('interaction.csv',interactions),('per_class_recall.csv',perclass),('compute.csv',compute)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 print(json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
