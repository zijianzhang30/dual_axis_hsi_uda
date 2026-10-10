import json,hashlib,time,shutil
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import confusion_matrix,cohen_kappa_score
BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2');FEATURE=BASE.parent/'tokenizer_stem_value_diagnostic_v1';OUT=BASE.parent/'houston_cross_domain_matching_diagnostic_v1';SEEDS=(2100,2101,2102);ALPHA=.25

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(1048576),b''):h.update(a)
 return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def normalize(a):
 a=np.asarray(a,np.float64);n=np.linalg.norm(a,axis=1,keepdims=True);assert np.isfinite(a).all() and (n>1e-12).all();return a/n

def nn_both(source,target,device='cuda'):
 s=torch.as_tensor(source,device=device,dtype=torch.float64);idx=[];cos=[];reverse=np.zeros(len(source),dtype=np.int64);best=np.full(len(source),-np.inf);start=0
 for x in np.array_split(target,(len(target)+1023)//1024):
  t=torch.as_tensor(x,device=device,dtype=torch.float64);sim=t@s.T;v,j=sim.max(1);idx.append(j.cpu().numpy());cos.append(v.cpu().numpy());v2,j2=sim.max(0);v2=v2.cpu().numpy();j2=j2.cpu().numpy()+start;better=v2>best;reverse[better]=j2[better];best[better]=v2[better];start+=len(x)
 j=np.concatenate(idx);v=np.concatenate(cos);mutual=reverse[j]==np.arange(len(target));assert mutual.sum()<=len(source);return j,v,reverse,mutual

def metric(y,p):
 pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=np.arange(7));rec=cm.diagonal()/cm.sum(1)*100;share=np.bincount(pred,minlength=7)/len(y)
 return {'OA':float((pred==y).mean()*100),'AA':float(rec.mean()),'Kappa':float(cohen_kappa_score(y,pred,labels=np.arange(7))),'recall':rec.tolist(),'prediction_shares':share.tolist(),'zero_recall_classes':[k+1 for k,v in enumerate(rec) if v==0],'collapse95':bool(share.max()>=.95),'confusion':cm.tolist()}

def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);files={};start=time.monotonic();plan={'dataset':'Houston','seeds':list(SEEDS),'checkpoint':'formal C+FullJointBN+source CE epoch200 all3','no_model_training_or_forward':True,'features':'reuse audited pure frozen T256 flattened and normalized CLS Z64;sourceval probe fitting did not alter features','reference':'originalsource2393train only,127valexcluded','target_candidates':'same native52901 nonzero-mask candidates inherited,not an expanded training permission;target classvalues not used until offline scoring','descriptors':['T L2cosine','Z L2cosine','TZ =equal mean T/Zcosine via separatelynormalized concat/sqrt2'],'matching':'top1 source for eachtarget;source reverse top1target;no distance threshold;stable firstindex tie break','mutual':'same top1 pairs intersect;fusion only retainedtargets,others originalprediction','fusion_alpha':ALPHA,'fusion':'(1-alpha)*existing C probability +alpha*onehot known nearestsource training label;allNN and mutualonly separately;no target labels or confidence filters','limits':'posthoc source-label transfer diagnostic,not learned cross-attention,not formal new network accuracy;Tflatten assumeswithin-model slot coordinate system;3seeds shareevaluationpixels','no_best_checkpoint_or_target_rule_selection':True,'no_auto_training_Pavia_HyRANK':True};dump(OUT/'frozen_plan.json',plan)
 mask=np.load(FEATURE/'data/source_probe_train_mask.npy');sy=np.load(FEATURE/'data/source_labels.npy')[mask].astype(np.int64);coords=np.load(FEATURE/'data/source_coords.npy')[mask];valcoords=np.load(FEATURE/'data/source_coords.npy')[~mask];assert mask.sum()==2393 and (~mask).sum()==127 and not set(map(tuple,coords))&set(map(tuple,valcoords));np.save(OUT/'source_reference_coords.npy',coords);np.save(OUT/'source_reference_labels.npy',sy)
 fixed=json.loads((FEATURE/'fixed_inputs.json').read_text());dump(OUT/'source_reference_audit.json',{'training_n':len(sy),'validation_excluded_n':len(valcoords),'source_class_counts':np.bincount(sy,minlength=7).tolist(),'mutual_coverage_upper_bound_percent':2393/52901*100,'reference_source_gt_allowed':True})
 for p in [FEATURE/'data/source_probe_train_mask.npy',FEATURE/'data/source_labels.npy',FEATURE/'data/source_coords.npy',FEATURE/'data/target_coords.npy',FEATURE/'fixed_inputs.json']:files[str(p)]=sha(p)
 # Pure numeric preflight;no dataset target GT.
 a=np.eye(3);j,v,rev,mut=nn_both(a,a);assert np.array_equal(j,np.arange(3)) and mut.all();j,v,rev,mut=nn_both(np.array([[1.,0.],[1.,0.]]),np.array([[1.,0.]]));assert j[0]==0 and rev.tolist()==[0,0]
 x=normalize(np.array([[1.,2.],[2.,1.]]));z=normalize(np.array([[3.,1.],[1.,3.]]));tz=np.concatenate((x,z),1)/np.sqrt(2);assert np.allclose(tz@tz.T,.5*(x@x.T+z@z.T));dump(OUT/'preflight.json',{'passed':True,'mutual_geometry':True,'deterministic_tie_break':True,'joint_equal_cosine_verified':True})
 allmatching={};manifests={}
 for seed in SEEDS:
  folder=OUT/('seed_%d'%seed);folder.mkdir();feature=FEATURE/('seed_%d'%seed);audit=json.loads((feature/'summary.json').read_text());assert audit['state_and_checkpoint_unchanged'] and max(audit['native_prediction_token_weight_audit']['target'].values())<1e-6
  ck=BASE/'houston'/('C_%d/checkpoints/epoch_200.pth'%seed);files[str(ck)]=sha(ck);assert files[str(ck)]==fixed['checkpoints'][str(seed)]['sha256'];cfg=json.loads((BASE/'houston'/('C_%d/config.json'%seed)).read_text());assert ahash(coords)==cfg['indices_sha256']['source'];assert ahash(valcoords)==cfg['indices_sha256']['val']
  arr={}
  for dom in ('source','target'):
   for stage in ('T','Z'):
    p=feature/(dom+'_'+stage+'.npy');files[str(p)]=sha(p);v=np.load(p,mmap_mode='r');arr[(dom,stage)]=normalize(v[mask] if dom=='source' else v);assert arr[(dom,stage)].shape==(2393 if dom=='source' else 52901,256 if stage=='T' else 64)
  path=feature/'target_logits.npy';files[str(path)]=sha(path);logits=np.load(path).astype(np.float64);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);np.save(folder/'original_probabilities.npy',p)
  allmatching[seed]={}
  for descriptor in ('T','Z','TZ'):
   if descriptor=='TZ':s=np.concatenate((arr[('source','T')],arr[('source','Z')]),1)/np.sqrt(2);t=np.concatenate((arr[('target','T')],arr[('target','Z')]),1)/np.sqrt(2)
   else:s,t=arr[('source',descriptor)],arr[('target',descriptor)]
   j,sim,rev,mut=nn_both(s,t);lab=sy[j];np.savez_compressed(folder/(descriptor+'_matching.npz'),source_reference_index=j,source_neighbor_label=lab,similarity=sim,source_reverse_target_index=rev,mutual_mask=mut)
   for policy in ('all','mutual'):
    apply=np.ones(len(t),bool) if policy=='all' else mut;mix=p.copy();mix[apply]*=1-ALPHA;mix[np.flatnonzero(apply),lab[apply]]+=ALPHA;assert np.isfinite(mix).all() and np.allclose(mix.sum(1),1);np.save(folder/(descriptor+'_'+policy+'_probabilities.npy'),mix)
   counts=np.bincount(j,minlength=len(sy));allmatching[seed][descriptor]={'mutual_count':int(mut.sum()),'mutual_coverage_percent':float(mut.mean()*100),'selected_source_class_distribution':(np.bincount(lab,minlength=7)/len(lab)).tolist(),'selected_unique_source_samples':int((counts>0).sum()),'max_single_source_selection_share':float(counts.max()/len(t)),'mean_similarity':float(sim.mean())};print('MATCH_FROZEN',seed,descriptor,'mutual',int(mut.sum()),flush=True)
  manifests[seed]={str(p):sha(p) for p in folder.glob('*.npy')};manifests[seed].update({str(p):sha(p) for p in folder.glob('*.npz')});dump(folder/'frozen_manifest.json',{'files':manifests[seed],'target_GT_used':False,'checkpoint_sha256':files[str(ck)]})
 dump(OUT/'unlabeled_matching_statistics.json',allmatching);dump(OUT/'input_manifest.json',files);dump(OUT/'GT_SCORING_BARRIER.json',{'all3_seed_matching_and_fusion_frozen':True,'rules_locked':True,'no_target_GT_used_for_matching':True,'all_output_hashes':manifests})
 # Offline target GT access begins here.
 y=np.load(FEATURE/'data/target_labels.npy').astype(int);targetcounts=np.bincount(y,minlength=7);sourceprior=np.bincount(sy,minlength=7)/len(sy);random_expected=float(sourceprior@(targetcounts/len(y))*100);rows=[];matchingrows=[];paired=[];full={}
 for seed in SEEDS:
  folder=OUT/('seed_%d'%seed);p=np.load(folder/'original_probabilities.npy');original=metric(y,p);pred=p.argmax(1);ok=pred==y;results={'original':original};matched={}
  old=np.load(BASE/'houston'/('C_%d/predictions_200.npz'%seed));assert np.array_equal(pred,old['predictions']);rows.append({'seed':seed,'descriptor':'original','policy':'none',**{k:original[k] for k in ('OA','AA','Kappa')}})
  for descriptor in ('T','Z','TZ'):
   nn=np.load(folder/(descriptor+'_matching.npz'));lab=nn['source_neighbor_label'];mut=nn['mutual_mask'];matched[descriptor]={}
   for policy,active in [('all',np.ones(len(y),bool)),('mutual',mut)]:
    cnt=np.bincount(y[active],minlength=7);per=[float((lab[active&(y==k)]==k).mean()*100) if cnt[k] else None for k in range(7)];cons=float((lab[active]==y[active]).mean()*100) if active.any() else None;q=np.load(folder/(descriptor+'_'+policy+'_probabilities.npy'));mt=metric(y,q);qp=q.argmax(1);corrected=int((~ok&(qp==y)).sum());broken=int((ok&(qp!=y)).sum());mr={'seed':seed,'descriptor':descriptor,'policy':policy,'coverage_percent':float(active.mean()*100),'matched_n':int(active.sum()),'match_OA':cons,'match_macro_present_classes':float(np.mean([v for v in per if v is not None])),'match_per_class_accuracy':per,'matched_target_class_counts':cnt.tolist(),'matching_accuracy_original_wrong':float((lab[active&~ok]==y[active&~ok]).mean()*100) if (active&~ok).any() else None,'matching_accuracy_original_correct':float((lab[active&ok]==y[active&ok]).mean()*100) if (active&ok).any() else None,'fusion_corrected':corrected,'fusion_broken':broken,'fusion_net':corrected-broken};matchingrows.append(mr);matched[descriptor][policy]=mr;results[descriptor+'_'+policy]=mt;rows.append({'seed':seed,'descriptor':descriptor,'policy':policy,**{k:mt[k] for k in ('OA','AA','Kappa')}});paired.append({'seed':seed,'group':descriptor+'_'+policy,**{k:mt[k]-original[k] for k in ('OA','AA','Kappa')}})
  full[str(seed)]={'classification_metrics':results,'matching_metrics':matched}
 aggregate=[]
 for descriptor,policy in [('original','none')]+[(d,p) for d in ('T','Z','TZ') for p in ('all','mutual')]:
  rr=[r for r in rows if r['descriptor']==descriptor and r['policy']==policy];aggregate.append({'descriptor':descriptor,'policy':policy,**{k:{'mean':float(np.mean([r[k] for r in rr])),'std_sample':float(np.std([r[k] for r in rr],ddof=1))} for k in ('OA','AA','Kappa')}})
 summary={'aggregate':aggregate,'paired':paired,'matching':matchingrows,'target_class_counts':targetcounts.tolist(),'random_source_pair_expected_agreement_percent':random_expected,'random_source_pair_class_macro_expected_percent':100/7,'source_training_reference_only':True,'target_GT_offline_only':True,'no_auto_training_or_other_benchmarks':True,'elapsed_seconds':time.monotonic()-start,'limits':'diagnostic nearest-source label transfer,not neural feature interaction;MNN sparse easy subset may have selection bias;same evaluatedpixels all3seeds;Tflatten within-model slot correspondence assumed'};dump(OUT/'summary.json',summary);dump(OUT/'per_seed_details.json',full)
 for name,rr in [('endpoints.csv',rows),('paired_deltas.csv',paired),('matching.csv',matchingrows)]:
  import csv
  with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 assert all(sha(p)==h for p,h in files.items());snap=OUT/'code_snapshot';snap.mkdir();shutil.copy2(__file__,snap/'run.py');dump(OUT/'status.json',{'status':'completed_no_training_no_followups','checkpoints_and_features_unchanged':True});print('MATCHING_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
