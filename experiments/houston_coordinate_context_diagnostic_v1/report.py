import json,csv,numpy as np
from sklearn.metrics import confusion_matrix,roc_auc_score
import common as c

def metrics(y,p):
 cm=confusion_matrix(y,p,labels=np.arange(7));n=cm.sum(1);rec=np.divide(np.diag(cm),n,out=np.zeros(7,float),where=n>0)
 return {'OA':float((p==y).mean()*100),'AA':float(rec.mean()*100),'recall':(rec*100).tolist(),'class_counts':n.tolist(),'confusion':cm.tolist()}
def auc(y,s):return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def main():
 manifests=[json.loads((c.OUT/('seed_%d/frozen_manifest.json'%s)).read_text()) for s in c.SEEDS]
 assert all(m['model_state_unchanged'] and m['BN_buffers_bitwise_unchanged'] and m['no_target_GT_accessed'] for m in manifests)
 for m in manifests:
  for p,h in m['hashes'].items():assert c.sha(p)==h,p
 c.dump(c.OUT/'GT_SCORING_BARRIER.json',{'all3_probe_configs_and_target_predictions_frozen':True,'frozen_manifests':{str(s):c.sha(c.OUT/('seed_%d/frozen_manifest.json'%s)) for s in c.SEEDS},'target_label_read_starts_after_this_barrier':True})
 ids=np.load(c.OUT/'data/target_historical_indices.npy');target_y=np.load(c.DIAG/'data/target_labels.npy')[ids];source_y=np.load(c.OUT/'data/source_labels.npy');val=~np.load(c.OUT/'data/source_train_mask.npy');results=[];classes=[]
 for seed in c.SEEDS:
  folder=c.OUT/('seed_%d'%seed);z=np.load(folder/'target_predictions.npz');p=z['probabilities'];pred=z['per_view_predictions'];s0=z['S0'];s1=z['S1'];a=metrics(target_y,s0);b=metrics(target_y,s1);wrong=s0!=target_y;fixed=wrong&(s1==target_y);broken=~wrong&(s1!=target_y);viewwrong=pred!=target_y[:,None];oracle=(~viewwrong).any(1)
  probmean=p.mean(1);js=(p*np.log(np.maximum(p,1e-12)/np.maximum(probmean[:,None,:],1e-12))).sum(-1).mean(1)
  votes=np.stack([(pred==k).sum(1) for k in range(7)],axis=1);disagreement=1-votes.max(1)/5.;pair=[]
  for i in range(5):
   for j in range(i+1,5):
    u=viewwrong[:,i].astype(float);v=viewwrong[:,j].astype(float);corr=float(np.corrcoef(u,v)[0,1]) if u.std()>0 and v.std()>0 else None
    pair.append({'view_i':i,'view_j':j,'error_correlation_phi':corr,'wrong_both_fraction':float((viewwrong[:,i]&viewwrong[:,j]).mean()),'prediction_agreement':float((pred[:,i]==pred[:,j]).mean())})
  row={'seed':seed,'S0':a,'S1':b,'delta_OA_pp':b['OA']-a['OA'],'delta_AA_pp':b['AA']-a['AA'],'Oracle_any_correct_OA':float(oracle.mean()*100),'Oracle_perclass_recall':[float(oracle[target_y==k].mean()*100) if (target_y==k).any() else None for k in range(7)],'S0_wrong':int(wrong.sum()),'S0_correct':int((~wrong).sum()),'corrected':int(fixed.sum()),'broken':int(broken.sum()),'corrected_fraction_of_S0_wrong':float(fixed.sum()/wrong.sum()),'broken_fraction_of_S0_correct':float(broken.sum()/(~wrong).sum()),'net_corrected':int(fixed.sum()-broken.sum()),'all5_views_wrong_fraction':float(viewwrong.all(1).mean()),'pairwise_error_diagnostics':pair,'S0_error_disagreement_AUROC':auc(wrong,disagreement),'S0_error_probability_JS_AUROC':auc(wrong,js),'disagreement_mean_S0_correct':float(disagreement[~wrong].mean()),'disagreement_mean_S0_wrong':float(disagreement[wrong].mean()),'JS_mean_S0_correct':float(js[~wrong].mean()),'JS_mean_S0_wrong':float(js[wrong].mean()),'zero_recall_S0':[k+1 for k,v in enumerate(a['recall']) if v==0],'zero_recall_S1':[k+1 for k,v in enumerate(b['recall']) if v==0]}
  sv=np.load(folder/'source_predictions.npz');row['source_val_S0']=metrics(source_y[val],sv['S0'][val]);row['source_val_S1']=metrics(source_y[val],sv['S1'][val]);row['compute']=manifests[list(c.SEEDS).index(seed)]['encoding_seconds_per_view'];row['probe_fit']=json.loads((folder/'probe_config.json').read_text())
  for k in range(7):
   mask=target_y==k;classes.append({'seed':seed,'class':k+1,'target_count':int(mask.sum()),'S0_recall':a['recall'][k],'S1_recall':b['recall'][k],'delta_recall':b['recall'][k]-a['recall'][k],'Oracle_any_correct_recall':row['Oracle_perclass_recall'][k],'corrected':int((fixed&mask).sum()),'broken':int((broken&mask).sum())})
  np.savez_compressed(folder/'offline_sample_diagnostics.npz',labels=target_y,S0_wrong=wrong,corrected=fixed,broken=broken,all_view_errors=viewwrong,disagreement=disagreement,JS=js,Oracle_any_correct=oracle)
  c.dump(folder/'offline_report.json',row);results.append(row)
 agg={}
 for name,fn in [('S0_OA',lambda r:r['S0']['OA']),('S0_AA',lambda r:r['S0']['AA']),('S1_OA',lambda r:r['S1']['OA']),('S1_AA',lambda r:r['S1']['AA']),('OA_delta_pp',lambda r:r['delta_OA_pp']),('AA_delta_pp',lambda r:r['delta_AA_pp']),('Oracle_any_correct_OA',lambda r:r['Oracle_any_correct_OA'])]:
  v=[fn(r) for r in results];agg[name]={'values':v,'mean':float(np.mean(v)),'std_sample':float(np.std(v,ddof=1))}
 summary={'aggregate':agg,'per_seed':results,'cohort':json.loads((c.OUT/'cohort.json').read_text()),'target_class_counts':np.bincount(target_y,minlength=7).tolist(),'main_network_not_trained_or_modified':True,'source_probe_fitted_on_five_views':True,'not_original_classifier_formal_OA':True,'Oracle_not_method_accuracy':True,'five_encodings_extra_context_union273_vs169':True,'development_task_not_independent_validation':True,'no_automatic_followups':True}
 c.dump(c.OUT/'summary.json',summary)
 with (c.OUT/'per_class_recall.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
 endpoints=[{'seed':r['seed'],'S0_OA':r['S0']['OA'],'S0_AA':r['S0']['AA'],'S1_OA':r['S1']['OA'],'S1_AA':r['S1']['AA'],'delta_OA':r['delta_OA_pp'],'delta_AA':r['delta_AA_pp'],'corrected':r['corrected'],'broken':r['broken'],'Oracle_any_correct':r['Oracle_any_correct_OA']} for r in results]
 with (c.OUT/'endpoints.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(endpoints[0]));w.writeheader();w.writerows(endpoints)
 c.verify();print(json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
