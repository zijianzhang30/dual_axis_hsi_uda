import csv,json
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from common import *
def metrics(z,y):
 z=np.asarray(z,float);pred=z.argmax(1);cm=confusion_matrix(y,pred,labels=np.arange(7));n=cm.sum(1);rec=np.divide(np.diag(cm),n,out=np.zeros(7),where=n>0)*100;counts=np.bincount(pred,minlength=7);nll=logsumexp(z,axis=1)-z[np.arange(len(y)),y]
 return {'OA':float((pred==y).mean()*100),'AA':float(rec[n>0].mean()),'Kappa':float(cohen_kappa_score(y,pred,labels=np.arange(7))),'NLL':float(nll.mean()),'per_class_recall':rec.tolist(),'class_n':n.tolist(),'prediction_class_counts':counts.tolist(),'prediction_class_ratio':(counts/len(y)).tolist(),'zero_recall_classes':[int(c+1) for c in range(7) if n[c] and rec[c]==0],'single_class_collapse_95pct':bool(counts.max()/len(y)>=.95),'confusion':cm.tolist()}
def main():
 verify();man=[];probes={}
 for seed in SEEDS:
  f=OUT/('seed_%d'%seed);m=json.loads((f/'frozen_manifest.json').read_text());assert m['probes']==14 and not m['target_GT_read'] and m['main_network_forwards']==0;assert all(sha(p)==h for p,h in m['files'].items());man.append(m);probes[seed]=json.loads((f/'probes.json').read_text())
 # Freeze predictions of all42 readouts before the first access to target class values.
 dump(OUT/'OFFLINE_SCORING_BARRIER.json',{'all42_probe_artifacts_frozen':True,'source_only_scalers_PCA_and_linear_fit':True,'feature_and_hyperparameters_locked':True,'target_GT_not_used_for_fit':True,'files':{p:h for m in man for p,h in m['files'].items()}})
 provenance=json.loads((BASE/'provenance.json').read_text());p=BASE/'data/target_labels.npy';assert sha(p)==provenance['data_sha256']['target_labels.npy'];yt=np.load(p);ys=np.load(BASE/'data/source_labels.npy');train=np.load(BASE/'data/source_probe_train_mask.npy');rows=[];pairs=[];preds={}
 for seed in SEEDS:
  preds[seed]={}
  for r in probes[seed]:
   a=np.load(r['path']);preds[seed][r['name']]=a['target_predictions'];values={'source_train':metrics(a['source_logits'][train],ys[train]),'source_val':metrics(a['source_logits'][~train],ys[~train]),'target':metrics(a['target_logits'],yt)};rows.append({'seed':seed,**r,'metrics':values})
  def pair(candidate,baseline):
   ca=next(r for r in rows if r['seed']==seed and r['name']==candidate);ba=next(r for r in rows if r['seed']==seed and r['name']==baseline);pc=preds[seed][candidate];pb=preds[seed][baseline];wrong=pb!=yt;newright=pc==yt
   d={'seed':seed,'candidate':candidate,'baseline':baseline,'domains':{domain:{k:ca['metrics'][domain][k]-ba['metrics'][domain][k] for k in ('OA','AA','Kappa','NLL')} for domain in ('source_train','source_val','target')},'target_corrected':int((wrong&newright).sum()),'target_broken':int((~wrong&~newright).sum()),'target_net_correct':int((pc==yt).sum()-(pb==yt).sum()),'per_class_recall_delta':[x-y for x,y in zip(ca['metrics']['target']['per_class_recall'],ba['metrics']['target']['per_class_recall'])],'per_class_corrected':[int((wrong&newright&(yt==c)).sum()) for c in range(7)],'per_class_broken':[int((~wrong&~newright&(yt==c)).sum()) for c in range(7)],'new_zero_recall_classes':sorted(set(ca['metrics']['target']['zero_recall_classes'])-set(ba['metrics']['target']['zero_recall_classes']))};pairs.append(d)
  pair('STRUCT','RAW')
  for st in ('T','Z'):
   pair(st+'_STRUCT',st);pair(st+'_RAW',st);pair(st+'_STRUCT',st+'_RAW');pair(st+'_PCA_STRUCT',st);pair(st+'_PCA_RAW',st);pair(st+'_PCA_STRUCT',st+'_PCA_RAW');pair(st+'_PCA_STRUCT',st+'_PCA_only')
 names=[r['name'] for r in probes[2100]];agg={}
 for name in names:
  r=[x for x in rows if x['name']==name];agg[name]={'dimensions':r[0]['dimensions'],'stored_linear_parameters':r[0]['stored_linear_parameters'],'domains':{domain:{key:{'mean':float(np.mean([x['metrics'][domain][key] for x in r])),'sample_std':float(np.std([x['metrics'][domain][key] for x in r],ddof=1))} for key in ('OA','AA','Kappa','NLL')} for domain in ('source_train','source_val','target')},'target_per_class_recall_mean':np.mean([x['metrics']['target']['per_class_recall'] for x in r],axis=0).tolist(),'per_seed':{str(x['seed']):x['metrics'] for x in r}}
 paired={}
 for candidate,baseline in sorted(set((p['candidate'],p['baseline']) for p in pairs)):
  rr=[p for p in pairs if p['candidate']==candidate and p['baseline']==baseline];paired[candidate+' minus '+baseline]={'target_mean_delta':{key:float(np.mean([p['domains']['target'][key] for p in rr])) for key in ('OA','AA','Kappa','NLL')},'source_val_mean_delta':{key:float(np.mean([p['domains']['source_val'][key] for p in rr])) for key in ('OA','AA','Kappa','NLL')},'seeds_both_OA_AA_positive':sum(p['domains']['target']['OA']>0 and p['domains']['target']['AA']>0 for p in rr),'per_seed':rr}
 def passed(c,b):return paired[c+' minus '+b]['seeds_both_OA_AA_positive']==3
 standalone=passed('STRUCT','RAW');full={st:passed(st+'_STRUCT',st) and passed(st+'_STRUCT',st+'_RAW') for st in ('T','Z')};matched={st:passed(st+'_PCA_STRUCT',st) and passed(st+'_PCA_STRUCT',st+'_PCA_RAW') for st in ('T','Z')};go=standalone and all(full.values()) and all(matched.values())
 result={'aggregate':agg,'paired':paired,'gate':{'structure_standalone_beats_raw_all3_OA_AA':standalone,'full_fusion_positive':full,'equal_total_dimension_positive':matched,'candidate_evidence_all_conditions':go,'automatic_followup':False},'limitations':'Frozen probes not official network results. Deterministic spectral transform adds no input information; tests readout accessibility and regularization. Only one predeclared index-local shape descriptor, not all spectral structures. No reliable wavelength/retained-band provenance. Raw/STRUCT standalone repeated seeds are identical data/split/probe fits, not independent variance estimates. Features,scalers,PCA all source-only fitted; target GT offline only. Historical target-nonzero candidate mask reused. Source-val127 and overlapping source/target patches limit independence; Houston development benchmark.'};dump(OUT/'summary.json',result);dump(OUT/'scored_probes.json',rows);dump(OUT/'paired_deltas.json',pairs)
 table=[];classes=[]
 for r in rows:
  for domain,v in r['metrics'].items():
   table.append({'seed':r['seed'],'probe':r['name'],'domain':domain,'dimensions':r['dimensions'],'stored_parameters':r['stored_linear_parameters'],**{k:v[k] for k in ('OA','AA','Kappa','NLL')},'zero_recall_classes':str(v['zero_recall_classes']),'collapse95':v['single_class_collapse_95pct']})
   for c in range(7):classes.append({'seed':r['seed'],'probe':r['name'],'domain':domain,'class':c+1,'n':v['class_n'][c],'recall':v['per_class_recall'][c],'predicted_n':v['prediction_class_counts'][c]})
 for name,rr in [('probe_results_NOT_FORMAL_OA.csv',table),('per_class_recall.csv',classes)]:
  with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 verify();dump(OUT/'status.json',{'status':'completed_frozen_spectral_probe_diagnostic','new_main_network_training':False,'followup_started':False});print(json.dumps({'gate':result['gate'],'target':{name:agg[name]['domains']['target'] for name in names},'paired':{k:v['target_mean_delta'] for k,v in paired.items()}},indent=2),flush=True)
if __name__=='__main__':main()
