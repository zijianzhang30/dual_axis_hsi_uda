import json,numpy as np,torch,csv
import study as s

def main():
 s.verify();torch.set_num_threads(2);c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',2100);dataset=ls[-1].dataset;y=np.asarray(dataset.labels).copy();dataset.labels=np.zeros_like(dataset.labels);dataset.label=np.zeros_like(dataset.label);loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(87100));m=m.cuda().eval();records={}
 for arm in ('Z','T'):
  f=s.OUT/('CDAN_'+arm+'_2100');pred=f/'offline_predictions';pred.mkdir(exist_ok=False);rs=[];mf=json.loads((f/'frozen_manifest.json').read_text());assert mf['completed_epochs']==100
  for r in mf['checkpoints']:
   if r['epoch']%10:continue
   assert s.sha(r['path'])==r['sha256'];m.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['model']);before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert before==c.state_hash(m);path=pred/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(path,logits=z,predictions=p);rs.append({'epoch':r['epoch'],'path':str(path),'sha256':s.sha(path)})
  s.dump(f/'offline_predictions_frozen.json',rs);records['CDAN_'+arm]=rs;print('PREDICTIONS_FROZEN',arm,flush=True)
 for group,old in [('CE','CE0_2100'),('LMMD','LMMD0_2100')]:
  old=s.B/'houston_shared_adapter_2x2_v1'/old/'offline_predictions_frozen.json';records[group]=[r for r in json.loads(old.read_text())['records'] if r['epoch']<=100]
 s.dump(s.OUT/'GT_SCORING_BARRIER.json',{'groups':list(records),'grid':list(range(10,101,10)),'all_predictions_frozen':True});ends=[];pc=[]
 for group,rs in records.items():
  assert [r['epoch'] for r in rs]==list(range(10,101,10));ev=[]
  for r in rs:
   assert s.sha(r['path'])==r['sha256'];p=np.load(r['path']);ev.append({'epoch':r['epoch'],'metrics':helper.metrics(y,p['predictions'],p['logits'],7)})
  s.dump(s.OUT/(group+'_scored_evaluations.json'),ev)
  for endpoint,r in [('fixed100',ev[-1]),('target_best',max(ev,key=lambda r:r['metrics']['oa_percent']))]:
   mt=r['metrics'];ends.append({'group':group,'seed':2100,'endpoint':endpoint,'epoch':r['epoch'],'OA':mt['oa_percent'],'AA':mt['aa_percent'],'Kappa':mt['kappa'],'zero_recall_classes':[k+1 for k,v in enumerate(mt['per_class_percent']) if v==0],'collapse95':bool(max(mt['prediction_shares'])>=.95),'prediction_shares':mt['prediction_shares']})
   for k,v in enumerate(mt['per_class_percent']):pc.append({'group':group,'endpoint':endpoint,'class':k+1,'recall':v,'prediction_share':mt['prediction_shares'][k]})
 paired=[]
 for ep in ('fixed100','target_best'):
  ee={r['group']:r for r in ends if r['endpoint']==ep}
  for left,right in [('CDAN_T','CDAN_Z'),('CDAN_T','CE'),('CDAN_Z','CE'),('CDAN_T','LMMD')]:paired.append({'endpoint':ep,'comparison':left+'-'+right,**{k:ee[left][k]-ee[right][k] for k in ('OA','AA','Kappa')}})
 s.dump(s.OUT/'summary.json',{'seed':2100,'exploratory_only':True,'fixed_endpoint':100,'endpoints':ends,'paired':paired,'no_mean_std_from_single_seed':True,'no_auto_expansion':True,'interpretation':'T vs Z confounded by direct Transformer gradient coverage; improvement alone cannot prove CLS information loss or original contribution.'})
 for name,rows in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',pc)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 s.verify();print('HOUSTON_CDAN_COMPLETE',flush=True)
if __name__=='__main__':main()
