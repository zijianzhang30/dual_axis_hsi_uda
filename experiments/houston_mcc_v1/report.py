import json,csv,numpy as np,torch
import study as s

def main():
 torch.set_num_threads(2);records={};helpers={};labels={};computes=[]
 # Inference all3 snapshots; logits frozen before target GT scoring.
 for seed in s.SEEDS:
  c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',seed);d=ls[-1].dataset;labels[seed]=np.asarray(d.labels).copy();d.labels=np.zeros_like(d.labels);d.label=np.zeros_like(d.label);helpers[seed]=helper;m=m.cuda().eval();folder=s.OUT/('MCC_%d'%seed);mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==200
  loader=torch.utils.data.DataLoader(d,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+85000));pred=folder/'offline_predictions';pred.mkdir(exist_ok=False);rs=[]
  for item in mf['checkpoints']:
   if item['epoch']%10:continue
   assert s.sha(item['path'])==item['sha256'];m.load_state_dict(torch.load(item['path'],map_location='cpu',weights_only=True)['model']);before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert before==c.state_hash(m);f=pred/('epoch_%03d.npz'%item['epoch']);np.savez_compressed(f,logits=z,predictions=p);rs.append({'epoch':item['epoch'],'path':str(f),'sha256':s.sha(f)})
  s.dump(folder/'offline_predictions_frozen.json',rs);records[('MCC',seed)]=rs
  baseline=s.B/'houston_shared_adapter_2x2_v1'/('CE0_%d'%seed);records[('CE',seed)]=json.loads((baseline/'offline_predictions_frozen.json').read_text())['records']
  old=json.loads((s.baseline(seed)/'frozen_manifest.json').read_text());computes.append({'seed':seed,'parameters_CE_and_MCC':sum(p.numel() for p in m.parameters()),'added_parameters':0,'MCC_train_seconds':mf['train_seconds'],'CE_historical_train_seconds':old['train_seconds'],'timing_limitation':'different runs/GPU loads; MCC includes diagnostic autograd overhead,not pure loss overhead benchmark'})
  del m,ls;torch.cuda.empty_cache();print('MCC_PREDICTIONS_FROZEN',seed,flush=True)
 s.dump(s.OUT/'all_predictions_frozen_before_scoring.json',{'groups':['CE','MCC'],'seeds':list(s.SEEDS),'grid':list(range(10,201,10))})
 ends=[];perclass=[]
 for (group,seed),rs in records.items():
  assert [r['epoch'] for r in rs]==list(range(10,201,10));ev=[]
  for r in rs:
   assert s.sha(r['path'])==r['sha256'];z=np.load(r['path']);ev.append({'epoch':r['epoch'],'metrics':helpers[seed].metrics(labels[seed],z['predictions'],z['logits'],7)})
  s.dump(s.OUT/('%s_%d_scored_evaluations.json'%(group,seed)),ev)
  for endpoint,r in [('fixed200',ev[-1]),('target_best',max(ev,key=lambda r:r['metrics']['oa_percent']))]:
   m=r['metrics'];ends.append({'group':group,'seed':seed,'endpoint':endpoint,'epoch':r['epoch'],'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],'zero_recall_classes':[i+1 for i,v in enumerate(m['per_class_percent']) if v==0],'collapse95':max(m['prediction_shares'])>=.95,'prediction_class_shares':m['prediction_shares']})
   perclass.extend([{'group':group,'seed':seed,'endpoint':endpoint,'class':i+1,'recall':v,'prediction_share':m['prediction_shares'][i]} for i,v in enumerate(m['per_class_percent'])])
 agg=[];paired=[]
 for ep in ('fixed200','target_best'):
  for group in ('CE','MCC'):
   rs=[r for r in ends if r['group']==group and r['endpoint']==ep];agg.append({'endpoint':ep,'group':group,**{k:{'mean':float(np.mean([r[k] for r in rs])),'std_sample':float(np.std([r[k] for r in rs],ddof=1))} for k in ('OA','AA','Kappa')}})
  for seed in s.SEEDS:
   rr={r['group']:r for r in ends if r['endpoint']==ep and r['seed']==seed};paired.append({'endpoint':ep,'seed':seed,**{k+'_delta':rr['MCC'][k]-rr['CE'][k] for k in ('OA','AA','Kappa')}})
 fixed=[r for r in paired if r['endpoint']=='fixed200'];s.dump(s.OUT/'summary.json',{'aggregate':agg,'paired':paired,'endpoints':ends,'compute':computes,'predeclared_fixed200_mean_OA_AA_positive':bool(np.mean([r['OA_delta'] for r in fixed])>0 and np.mean([r['AA_delta'] for r in fixed])>0),'fixed200_both_positive_seed_count':sum(r['OA_delta']>0 and r['AA_delta']>0 for r in fixed),'interpretation':'Go additionally requires review of perseed severe degradation/class coverage;loss decrease/no collapse are not success','MCC_existing_method_not_our_innovation':True,'target_best_target_label_selection_separate':True,'no_auto_followups':True})
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',perclass)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 s.verify();print('MCC_RESULTS_COMPLETE',flush=True)
if __name__=='__main__':main()
