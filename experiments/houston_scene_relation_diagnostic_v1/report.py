import json,csv,numpy as np
from sklearn.metrics import confusion_matrix,cohen_kappa_score
import common as c

def metric(y,p):
 pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=np.arange(7));rec=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(7,float),where=cm.sum(1)>0)*100;shares=np.bincount(pred,minlength=7)/len(y)
 return {'n':len(y),'OA':float((pred==y).mean()*100),'AA':float(rec.mean()),'Kappa':float(cohen_kappa_score(y,pred,labels=np.arange(7))),'recall':rec.tolist(),'class_counts':np.bincount(y,minlength=7).tolist(),'prediction_shares':shares.tolist(),'zero_recall_classes':[i+1 for i,r in enumerate(rec) if r==0],'collapse95':bool(shares.max()>=.95),'confusion':cm.tolist()}

def main():
 c.verify();man={}
 for seed in c.SEEDS:
  mf=json.loads((c.OUT/('seed_%d/frozen_manifest.json'%seed)).read_text());assert mf['model_state_unchanged'] and not mf['target_GT_accessed'];assert all(c.sha(p)==h for p,h in mf['files'].items());man[str(seed)]=mf
 c.dump(c.OUT/'GT_SCORING_BARRIER.json',{'all3_seed_dense_predictions_frozen':True,'all3_group_rules_locked':True,'manifests':man,'no_model_training':True})
 # First target coordinate/GT access happens here,after graph/rules and outputs frozen.
 coords=np.load(c.DATA/'target_coords.npy')-6;y=np.load(c.DATA/'target_labels.npy').astype(int);gt=np.load(c.DATA/'target_gt.npy')[6:-6,6:-6];h,w=gt.shape;assert np.array_equal(gt[coords[:,0],coords[:,1]],y);flat=coords[:,0]*w+coords[:,1];n=np.load(c.OUT/'neighbors.npy');linked=np.load(c.OUT/'uniform_weights.npy')>0;gy=gt.flatten();neighbor_y=gy[n[flat]];labeled=linked[flat]&(neighbor_y>=0);boundary=(labeled&(neighbor_y!=y[:,None])).any(1);interior=(labeled&(neighbor_y==y[:,None])).all(1);unknown=~(boundary|interior)
 np.save(c.OUT/'offline_native_coords.npy',coords);np.save(c.OUT/'offline_boundary_mask.npy',boundary);np.save(c.OUT/'offline_strict_interior_mask.npy',interior);c.dump(c.OUT/'offline_cohort.json',{'native_n':len(y),'class_counts':np.bincount(y,minlength=7).tolist(),'boundary_n':int(boundary.sum()),'strict_interior_n':int(interior.sum()),'unknown_or_incomplete_n':int(unknown.sum()),'boundary_GT_offline_only':True,'graph_uses_GT':False})
 rows=[];paired=[];pc=[];diagnostics={}
 for seed in c.SEEDS:
  folder=c.OUT/('seed_%d'%seed);pred={};metrics={}
  for group in ('A','B','C'):
   p=np.load(folder/('probabilities_'+group+'.npy'),mmap_mode='r')[flat];pred[group]=p.argmax(1);metrics[group]=metric(y,p);rows.append({'seed':seed,'group':group,**{k:metrics[group][k] for k in ('OA','AA','Kappa','zero_recall_classes','collapse95')}})
   pc.extend([{'seed':seed,'group':group,'class':k+1,'n':metrics[group]['class_counts'][k],'recall':v,'prediction_share':metrics[group]['prediction_shares'][k]} for k,v in enumerate(metrics[group]['recall'])])
  old=c.B/'houston_shared_adapter_2x2_v1'/('CE0_%d/offline_predictions/epoch_200.npz'%seed);historical=np.load(old);original=np.load(folder/'full_scene_logits.npy',mmap_mode='r')[flat];mismatch=int((pred['A']!=historical['predictions']).sum());assert mismatch==0,('dense A differs from historical predictions,report before proceeding',seed,mismatch)
  subsets={};correct0=pred['A']==y
  for name,mask in [('all',np.ones(len(y),bool)),('boundary',boundary),('strict_interior',interior),('unknown_or_incomplete',unknown)]:
   subsets[name]={}
   for group in ('B','C'):
    ok=pred[group]==y;fixed=int((mask&~correct0&ok).sum());broken=int((mask&correct0&~ok).sum());subsets[name][group]={'n':int(mask.sum()),'corrected':fixed,'broken':broken,'net':fixed-broken,'OA':float(ok[mask].mean()*100) if mask.any() else None}
  for left,right in [('B','A'),('C','A'),('C','B')]:paired.append({'seed':seed,'comparison':left+'-'+right,**{k:metrics[left][k]-metrics[right][k] for k in ('OA','AA','Kappa')}})
  diagnostics[str(seed)]={'metrics':metrics,'corrected_broken_relative_A':subsets,'baseline_prediction_mismatches':mismatch,'baseline_logits_max_abs_difference':float(np.max(np.abs(original-historical['logits']))),'native_node_spectral_valid_fraction':float(np.load(c.OUT/'spectral_valid.npy').flatten()[flat].mean()),'class_corrected_broken':{group:[{'class':k+1,'n':int((y==k).sum()),'corrected':int(((y==k)&~correct0&(pred[group]==y)).sum()),'broken':int(((y==k)&correct0&(pred[group]!=y)).sum())} for k in range(7)] for group in ('B','C')}}
 c.dump(c.OUT/'per_seed_diagnostics.json',diagnostics);agg=[];deltas=[]
 for group in ('A','B','C'):
  rr=[r for r in rows if r['group']==group];agg.append({'group':group,**{k:{'mean':float(np.mean([r[k] for r in rr])),'std_sample':float(np.std([r[k] for r in rr],ddof=1))} for k in ('OA','AA','Kappa')}})
 for comp in ('B-A','C-A','C-B'):
  rr=[r for r in paired if r['comparison']==comp];deltas.append({'comparison':comp,**{k:{'mean':float(np.mean([r[k] for r in rr])),'std_sample':float(np.std([r[k] for r in rr],ddof=1))} for k in ('OA','AA','Kappa')},'both_positive_seeds':sum(r['OA']>0 and r['AA']>0 for r in rr)})
 s={'fixed_epoch':200,'aggregate':agg,'paired_per_seed':paired,'paired_summary':deltas,'no_target_best_selection':True,'baseline_predictions_exactly_match_historical_all3':True,'no_auto_followups':True,'scope':'frozen model one-pass postprocessing feasibility,not trained new network;extra physical-neighbor probability inference budget;no claim own innovation;image boundaries/labeled boundaries differ'};c.dump(c.OUT/'summary.json',s)
 for name,rr in [('endpoints.csv',rows),('paired_deltas.csv',paired),('per_class_recall.csv',pc)]:
  with (c.OUT/name).open('w') as f:writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
 c.verify();print(json.dumps(s,indent=2),flush=True)
if __name__=='__main__':main()
