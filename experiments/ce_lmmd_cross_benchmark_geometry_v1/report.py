import json,csv,numpy as np
from sklearn.metrics import confusion_matrix,cohen_kappa_score
import common as c

def unit(x):
 x=np.asarray(x,np.float64);n=np.linalg.norm(x,axis=-1,keepdims=True);return x/np.maximum(n,1e-12)
def metric(y,z):
 p=z.argmax(1);cm=confusion_matrix(y,p,labels=np.arange(7));rec=cm.diagonal()/cm.sum(1)*100
 return {'OA':float((p==y).mean()*100),'AA':float(rec.mean()),'Kappa':float(cohen_kappa_score(y,p,labels=np.arange(7))),'recall':rec.tolist(),'confusion':cm.tolist()}
def geometry(source,target,sy,ty):
 raw_s=np.asarray(source,np.float64);raw_t=np.asarray(target,np.float64);s=unit(raw_s);t=unit(raw_t);ss=np.stack([s[sy==k].sum(0) for k in range(7)]);ts=np.stack([t[ty==k].sum(0) for k in range(7)]);sc=unit(ss);tc=unit(ts);scores=t@sc.T;pred=scores.argmax(1);inter=tc@sc.T;rows=[]
 for k in range(7):
  a=sy==k;b=ty==k;others=[j for j in range(7) if j!=k];margin=scores[b,k]-scores[b][:,others].max(1);src=s[a];true=(src*unit(ss[k]-src)).sum(1);smargin=true-(src@sc[others].T).max(1);tgt=t[b];ttrue=(tgt*unit(ts[k]-tgt)).sum(1);tmargin=ttrue-(tgt@tc[others].T).max(1)
  rows.append({'class':k+1,'source_n':int(a.sum()),'target_n':int(b.sum()),'source_norm_mean':float(np.linalg.norm(raw_s[a],axis=1).mean()),'target_norm_mean':float(np.linalg.norm(raw_t[b],axis=1).mean()),'source_within_class_cosine_dispersion':float((1-(src*sc[k]).sum(1)).mean()),'target_within_class_cosine_dispersion':float((1-(tgt*tc[k]).sum(1)).mean()),'source_LOO_class_margin':float(smargin.mean()),'target_GT_LOO_class_margin':float(tmargin.mean()),'target_correct_source_margin':float(margin.mean()),'target_negative_source_margin_fraction':float((margin<0).mean()),'target_source_centroid_recall':float((pred[b]==k).mean()*100),'source_target_same_class_centroid_cosine':float(inter[k,k]),'target_class_centroid_correct_source_margin':float(inter[k,k]-inter[k,others].max()),'target_centroid_nearest_source_class':int(inter[k].argmax())+1})
 return {'classes':rows,'cross_class_centroid_cosine':inter.tolist(),'nearest_source_confusion':confusion_matrix(ty,pred,labels=np.arange(7)).tolist(),'global_unit_mean_distance':float(np.linalg.norm(s.mean(0)-t.mean(0))),'source_zero_norm_n':int((np.linalg.norm(raw_s,axis=1)<=1e-12).sum()),'target_zero_norm_n':int((np.linalg.norm(raw_t,axis=1)<=1e-12).sum()),'source_centroid_classification_OA':float((pred==ty).mean()*100),'source_centroid_classification_AA':float(np.mean([r['target_source_centroid_recall'] for r in rows]))}

def main():
 c.verify();manifests={};data={}
 for dataset in ('houston','pavia'):
  for seed in c.SEEDS:
   job=c.OUT/('%s_%d'%(dataset,seed));assert (job/'complete.json').exists();epochs=(20,200) if dataset=='houston' else (10,100)
   for objective in ('CE','LMMD'):
    for epoch in epochs:
     cell=job/('%s_%03d'%(objective,epoch));m=json.loads((cell/'frozen_manifest.json').read_text());assert m['state_unchanged'] and not m['target_GT_accessed'];assert all(c.sha(p)==h for p,h in m['files'].items());manifests[(dataset,seed,objective,epoch)]=m
 c.dump(c.OUT/'GT_SCORING_BARRIER.json',{'all24_cells_frozen':True,'cohorts_and_rules_locked':True,'no_training':True})
 rows=[];endpoints=[];results={};canonical=[]
 for key,m in manifests.items():
  dataset,seed,objective,epoch=key;job=c.OUT/('%s_%d'%(dataset,seed));sy=np.load(job/'source_labels.npy');ty=np.load(c.B/('tokenizer_stem_value_diagnostic_v1/data/target_labels.npy' if dataset=='houston' else 'pavia_protocol_transfer_v1/inputs/evaluation_labels_OFFLINE_ONLY.npy'));z=np.load(m['features']['target']['logits']);cl=metric(ty,z);endpoints.append({'dataset':dataset,'seed':seed,'objective':objective,'epoch':epoch,**{k:cl[k] for k in ('OA','AA','Kappa')}});rs={'classification':cl,'stages':{}}
  final=200 if dataset=='houston' else 100
  if epoch==final:
   if dataset=='houston':
    base=c.B/'houston_shared_adapter_2x2_v1'/('%s0_%d/offline_predictions/epoch_200.npz'%('CE' if objective=='CE' else 'LMMD',seed))
   else:base=c.folder(dataset,objective,seed)/'predictions_epoch100.npz'
   old=np.load(base);assert np.array_equal(z.argmax(1),old['predictions']),('formal predictions differ',key);canonical.append({'dataset':dataset,'seed':seed,'objective':objective,'argmax_matches_formal':True,'logits_max_error':float(np.abs(z-old['logits']).max())})
  prob=np.exp(z.astype(np.float64)-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
  rs['soft_membership_purity_eval_only']=[float(prob[ty==k,k].sum()/prob[:,k].sum()) for k in range(7)]
  for stage in c.STAGES:
   sf=np.load(m['features']['source'][stage],mmap_mode='r');tf=np.load(m['features']['target'][stage],mmap_mode='r');geo=geometry(sf,tf,sy,ty);rs['stages'][stage]=geo
   rows.extend([{'dataset':dataset,'seed':seed,'objective':objective,'epoch':epoch,'stage':stage,**r} for r in geo['classes']])
  results['%s_%d_%s_%d'%key]=rs
 c.dump(c.OUT/'geometry_details.json',results);c.dump(c.OUT/'formal_prediction_audit.json',canonical);paired=[]
 keys=['source_norm_mean','target_norm_mean','source_within_class_cosine_dispersion','target_within_class_cosine_dispersion','source_LOO_class_margin','target_GT_LOO_class_margin','target_correct_source_margin','target_negative_source_margin_fraction','target_source_centroid_recall','source_target_same_class_centroid_cosine','target_class_centroid_correct_source_margin']
 for dataset in ('houston','pavia'):
  for seed in c.SEEDS:
   for epoch in ((20,200) if dataset=='houston' else (10,100)):
    for stage in c.STAGES:
     for cls in range(1,8):
      rr={r['objective']:r for r in rows if r['dataset']==dataset and r['seed']==seed and r['epoch']==epoch and r['stage']==stage and r['class']==cls};paired.append({'dataset':dataset,'seed':seed,'epoch':epoch,'stage':stage,'class':cls,**{k:rr['LMMD'][k]-rr['CE'][k] for k in keys}})
 aggregate=[]
 for dataset in ('houston','pavia'):
  for epoch in ((20,200) if dataset=='houston' else (10,100)):
   for stage in c.STAGES:
    rr=[r for r in paired if r['dataset']==dataset and r['epoch']==epoch and r['stage']==stage];byseed=[{k:float(np.mean([r[k] for r in rr if r['seed']==s])) for k in keys} for s in c.SEEDS];aggregate.append({'dataset':dataset,'epoch':epoch,'stage':stage,**{k:{'classbalanced_mean_delta':float(np.mean([r[k] for r in byseed])),'per_seed_classbalanced_delta':[r[k] for r in byseed]} for k in keys}})
 for name,rr in [('per_class_geometry.csv',rows),('paired_CE_to_LMMD.csv',paired),('classifier_endpoints.csv',endpoints)]:
  with (c.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 c.dump(c.OUT/'summary.json',{'aggregate_paired_geometry':aggregate,'classifier_endpoints':endpoints,'formal_prediction_audit_passed':True,'same_protocol_CE_to_LMMD_only':True,'existing_gradient_audit_reused':str(c.B/'lmmd_houston_pavia_execution_audit_v1'),'limitations':'Frozen early/end checkpoint association,not causal gradient reconstruction;GT classcentroids diagnostic only. Crossdatasets differ normalization/sampling/optimizer/BNview budget. Global mean gap sensitive to class priors;lower gap not success. No class-aware rule fitting,probe retraining or new module.'});c.verify();print('GEOMETRY_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
