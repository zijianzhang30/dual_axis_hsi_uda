import json,csv,numpy as np,torch
import study as s

def metrics_endpoints(records,seed,group,dataset,scope,helper,labels):
 rs=[]
 for r in records:
  z=np.load(r['path']);assert s.sha(r['path'])==r['sha256'];rs.append({'epoch':r['epoch'],'metrics':helper.metrics(labels,z['predictions'],z['logits'],12 if dataset=='HyRANK' else 7)})
 endpoints=[];perclass=[]
 for ep,r in [('fixed200',rs[-1]),('target_best',max(rs,key=lambda x:x['metrics']['oa_percent']))]:
  m=r['metrics'];endpoints.append({'dataset':dataset,'scope':scope,'group':group,'seed':seed,'endpoint':ep,'epoch':r['epoch'],'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],'zero_recall_classes':[k+1 for k,v in enumerate(m['per_class_percent']) if v==0],'collapse95':max(m['prediction_shares'])>=.95})
  perclass.extend([{'dataset':dataset,'scope':scope,'group':group,'seed':seed,'endpoint':ep,'class':k+1,'recall':v} for k,v in enumerate(m['per_class_percent'])])
 return endpoints,perclass,rs

def main():
 torch.set_num_threads(2);ends=[];classes=[]
 c,ns=s.environment();args=c.options(2100);ds=ns['loaders'](args)[0][-1].dataset;labels={'native':np.asarray(ds.labels),'full':s.geometry().full_geometry(ds)[-1]}
 for group in ('CE','LMMD'):
  for seed in (2100,2101,2102):
   origin=(s.B/('hyrank_ce_eq9_off_2100_v1' if group=='CE' else 'hyrank_lmmd_eq9_off_2100_v1')/'C_2100') if seed==2100 else s.TOP/group/('C_%d'%seed)
   manifest=json.loads((origin/'frozen_manifest.json').read_text());assert manifest['completed_epochs']==200
   for scope in ('native','full'):
    records=[{'epoch':r['epoch'],'path':r[scope+'_prediction_path'],'sha256':r[scope+'_sha256']} for r in manifest['evaluations']];assert [r['epoch'] for r in records]==list(range(10,201,10))
    e,p,rs=metrics_endpoints(records,seed,group,'HyRANK',scope,c.helper,labels[scope]);ends+=e;classes+=p
    if seed!=2100:s.dump(origin/('offline_scored_%s.json'%scope),rs)
 # Houston new .001 checkpoints: freeze all predictions before using target class values.
 import houston_common as hc
 m,ls,helper,n,audit=hc.setup('houston','C',2100);m=m.cuda().eval();d=ls[-1].dataset;gt=np.asarray(d.labels).copy();d.labels=np.zeros_like(d.labels);d.label=np.zeros_like(d.label)
 loader=torch.utils.data.DataLoader(d,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(87100))
 origin=s.TOP/'Houston_lr001/houston/C_LMMD_2100';mf=json.loads((origin/'frozen_manifest.json').read_text());pred=origin/'offline_predictions';pred.mkdir(exist_ok=False);recs=[]
 for r in mf['checkpoints']:
  if r['epoch']%10:continue
  assert s.sha(r['path'])==r['sha256'];m.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['model']);before=hc.state_hash(m);logits,pr=helper.predict(m,loader,torch.device('cuda:0'));assert hc.state_hash(m)==before
  p=pred/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(p,logits=logits,predictions=pr);recs.append({'epoch':r['epoch'],'path':str(p),'sha256':s.sha(p)})
 s.dump(origin/'offline_predictions_frozen.json',recs)
 e,p,rs=metrics_endpoints(recs,2100,'LMMD_lr001','Houston','native',helper,gt);ends+=e;classes+=p;s.dump(origin/'offline_scored_evaluations.json',rs)
 baseline=s.B/'houston_shared_adapter_2x2_v1/LMMD0_2100';recs=json.loads((baseline/'offline_predictions_frozen.json').read_text())['records'];e,p,rs=metrics_endpoints(recs,2100,'LMMD_lr01','Houston','native',helper,gt);ends+=e;classes+=p
 agg=[];paired=[]
 for scope in ('native','full'):
  for ep in ('fixed200','target_best'):
   for group in ('CE','LMMD'):
    rr=[r for r in ends if r['dataset']=='HyRANK' and r['scope']==scope and r['endpoint']==ep and r['group']==group];assert len(rr)==3
    agg.append({'dataset':'HyRANK','scope':scope,'endpoint':ep,'group':group,**{k:{'mean':float(np.mean([r[k] for r in rr])),'std_sample':float(np.std([r[k] for r in rr],ddof=1))} for k in ('OA','AA','Kappa')}})
   for seed in (2100,2101,2102):
    rr={r['group']:r for r in ends if r['dataset']=='HyRANK' and r['scope']==scope and r['endpoint']==ep and r['seed']==seed};paired.append({'dataset':'HyRANK','scope':scope,'endpoint':ep,'seed':seed,'comparison':'LMMD-CE',**{k+'_delta':rr['LMMD'][k]-rr['CE'][k] for k in ('OA','AA','Kappa')}})
 for ep in ('fixed200','target_best'):
  rr={r['group']:r for r in ends if r['dataset']=='Houston' and r['endpoint']==ep};paired.append({'dataset':'Houston','scope':'native','endpoint':ep,'seed':2100,'comparison':'lr001-lr01',**{k+'_delta':rr['LMMD_lr001'][k]-rr['LMMD_lr01'][k] for k in ('OA','AA','Kappa')}})
 s.dump(s.TOP/'summary.json',{'HyRANK_aggregate':agg,'endpoints':ends,'paired':paired,'Houston_is_single_seed_lr_only_diagnostic':True,'HyRANK_seed2100_reused':True,'new_trainings':5,'no_auto_followups':True})
 for name,rr in [('endpoints.csv',ends),('per_class_recall.csv',classes),('paired_deltas.csv',paired)]:
  with (s.TOP/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 print('ALL_FIVE_RESULTS_REPORTED',flush=True)
if __name__=='__main__':main()
