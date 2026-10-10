import pathlib,json,csv,hashlib,shutil,time
import numpy as np
B=pathlib.Path('/nas1/zhangzj26/dual_axis_hsi_uda/results');OUT=B/'houston_TZ_correspondence_transition_v1';MATCH=B/'houston_cross_domain_matching_diagnostic_v1';FEAT=B/'tokenizer_stem_value_diagnostic_v1';CENT=B/'tokenizer_class_correspondence_houston_v1';SEEDS=(2100,2101,2102)
def dump(p,v):pathlib.Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def csvout(name,rows):
 with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def transition(t,z,y,mask):
 tc=t==y;zc=z==y;n=int(mask.sum());loss=int((mask&tc&~zc).sum());rescue=int((mask&~tc&zc).sum())
 return {'n':n,'both_correct':int((mask&tc&zc).sum()),'T_correct_Z_wrong':loss,'T_wrong_Z_correct':rescue,'both_wrong':int((mask&~tc&~zc).sum()),'T_accuracy':float(tc[mask].mean()*100) if n else None,'Z_accuracy':float(zc[mask].mean()*100) if n else None,'Z_minus_T_pp':float((rescue-loss)/n*100) if n else None,'Z_rescues_minus_losses':rescue-loss}

def main():
 OUT.mkdir(exist_ok=False);plan={'read_only':True,'dataset':'Houston','seeds':list(SEEDS),'checkpoint':'formal C+CE epoch200','reuse_only':True,'no_new_matching_or_probes_or_forward':True,'methods':['source_train_1NN_cosine_existing','source_train_centroid_cosine_existing','source_only_standardized_logistic_probe_existing'],'groups':['all_target','original_classifier_correct','original_classifier_wrong','original_final_C6_to_C5'],'classes':'all7,reporteachseed andmean;no target class-basedmodelrules','events':'TcorrectZwrong vsTwrongZcorrect using same offline target GT','interpretation':'different T/Z dimensional geometries and probes;changed neighbors not causal information destruction;conditioning on final errors creates selection effects;no trainingfollowup'};dump(OUT/'frozen_plan.json',plan);files={};rows=[];flows=[];all_events={};source_partition=np.load(FEAT/'data/source_probe_train_mask.npy');source_coords=np.load(FEAT/'data/source_coords.npy')[source_partition];assert len(source_coords)==2393
 for p in [FEAT/'data/source_probe_train_mask.npy',FEAT/'data/source_coords.npy',MATCH/'frozen_plan.json',CENT/'CONCLUSIONS.json',CENT/'FINAL_INTERPRETATION.txt']:files[str(p)]=sha(p)
 # Existing target labels only used for offline descriptive analysis after plan lock.
 y=np.load(FEAT/'data/target_labels.npy').astype(int);coords=np.load(FEAT/'data/target_coords.npy');files[str(FEAT/'data/target_labels.npy')]=sha(FEAT/'data/target_labels.npy');files[str(FEAT/'data/target_coords.npy')]=sha(FEAT/'data/target_coords.npy')
 for seed in SEEDS:
  folder=OUT/('seed_%d'%seed);folder.mkdir();origpath=FEAT/('seed_%d/target_logits.npy'%seed);files[str(origpath)]=sha(origpath);original=np.load(origpath).argmax(1);ok=original==y;methods={};events={};checkpoint=B/'tokenizer_mechanism_v2/houston'/('C_%d/checkpoints/epoch_200.pth'%seed);files[str(checkpoint)]=sha(checkpoint)
  cfg=json.load(open(B/'tokenizer_mechanism_v2/houston'/('C_%d/config.json'%seed)));assert hashlib.sha256(np.ascontiguousarray(source_coords).tobytes()).hexdigest()==cfg['indices_sha256']['source']
  for stage in ('T','Z'):
   npth=MATCH/('seed_%d/%s_matching.npz'%(seed,stage));cpth=CENT/('seed_%d/C_%s_source_centroid_scores.npz'%(seed,stage));ppth=FEAT/('seed_%d/%s_source_probe.npz'%(seed,stage))
   for p in (npth,cpth,ppth):files[str(p)]=sha(p)
   methods.setdefault('nearest_neighbor',{})[stage]=np.load(npth)['source_neighbor_label'];methods.setdefault('class_centroid',{})[stage]=np.load(cpth)['target_scores'].argmax(1);methods.setdefault('source_probe',{})[stage]=np.load(ppth)['target_predictions']
  masks={'all_target':np.ones(len(y),bool),'original_classifier_correct':ok,'original_classifier_wrong':~ok,'original_final_C6_to_C5':(y==5)&(original==4)}
  for name,m in methods.items():
   t,z=m['T'],m['Z'];assert len(t)==len(z)==len(y);tc=t==y;zc=z==y;events[name]={'T_prediction':t,'Z_prediction':z,'T_correct_Z_wrong':tc&~zc,'T_wrong_Z_correct':~tc&zc};all_events[(seed,name)]=(tc&~zc,~tc&zc)
   for subset,mask in masks.items():
    for label,k in [('all',None)]+[(str(k+1),k) for k in range(7)]:
     selected=mask if k is None else mask&(y==k);r=transition(t,z,y,selected);rows.append({'seed':seed,'method':name,'subset':subset,'class':label,**r})
   for k in range(7):
    for j in range(7):
     if j==k:continue
     loss=(y==k)&tc&(z==j);gain=(y==k)&(t==j)&zc;flows.append({'seed':seed,'method':name,'true_class':k+1,'other_class':j+1,'T_correct_to_Z_other':int(loss.sum()),'T_other_to_Z_correct':int(gain.sum()),'lost_subset_original_classifier_wrong':int((loss&~ok).sum()),'rescued_subset_original_classifier_wrong':int((gain&~ok).sum())})
  # Strongest descriptive event: agreement across all3 readouts,not a tuned method.
  commonloss=np.logical_and.reduce([v['T_correct_Z_wrong'] for v in events.values()]);commongain=np.logical_and.reduce([v['T_wrong_Z_correct'] for v in events.values()]);events['all3_readouts_loss_intersection']=commonloss;events['all3_readouts_gain_intersection']=commongain;events['original_classifier_prediction']=original;events['target_coords_padded']=coords;events['offline_GT']=y
  np.savez_compressed(folder/'per_pixel_events.npz',**{key+'_'+sub:val for key,part in events.items() if isinstance(part,dict) for sub,val in part.items()},**{key:part for key,part in events.items() if not isinstance(part,dict)})
  print('TRANSITIONS_COMPLETE',seed,flush=True)
 aggregate=[]
 for method in ('nearest_neighbor','class_centroid','source_probe'):
  for subset in ('all_target','original_classifier_correct','original_classifier_wrong','original_final_C6_to_C5'):
   for cls in ('all',)+tuple(str(k) for k in range(1,8)):
    rr=[r for r in rows if r['method']==method and r['subset']==subset and r['class']==cls];valid=[r for r in rr if r['n']];row={'method':method,'subset':subset,'class':cls,'n_per_seed':[r['n'] for r in rr],'T_correct_Z_wrong_per_seed':[r['T_correct_Z_wrong'] for r in rr],'T_wrong_Z_correct_per_seed':[r['T_wrong_Z_correct'] for r in rr],'Z_net_gain_per_seed':[r['Z_rescues_minus_losses'] for r in rr],'Z_gain_seed_count':sum(r['Z_rescues_minus_losses']>0 for r in rr),'T_gain_seed_count':sum(r['Z_rescues_minus_losses']<0 for r in rr)}
    for key in ('T_accuracy','Z_accuracy','Z_minus_T_pp'):row[key+'_mean']=float(np.mean([r[key] for r in valid])) if valid else None
    aggregate.append(row)
 consistency=[]
 for method in ('nearest_neighbor','class_centroid','source_probe'):
  losses=[all_events[(seed,method)][0] for seed in SEEDS];gains=[all_events[(seed,method)][1] for seed in SEEDS];bothloss=np.logical_and.reduce(losses);bothgain=np.logical_and.reduce(gains)
  consistency.append({'method':method,'same_pixels_TcorrectZwrong_all3seeds':int(bothloss.sum()),'same_pixels_TwrongZcorrect_all3seeds':int(bothgain.sum()),'loss_all3_per_class':np.bincount(y[bothloss],minlength=7).tolist(),'gain_all3_per_class':np.bincount(y[bothgain],minlength=7).tolist()})
 csvout('transitions_per_seed.csv',rows);csvout('class_error_flows.csv',flows);csvout('aggregate.csv',aggregate);dump(OUT/'summary.json',{'aggregate':aggregate,'same_pixel_cross_seed_consistency':consistency,'previous_C6_centroid_findings_reused':json.load(open(CENT/'CONCLUSIONS.json')),'limitations':plan['interpretation'],'no_auto_new_network':True});dump(OUT/'input_hashes.json',files);assert all(sha(p)==h for p,h in files.items());snap=OUT/'code_snapshot';snap.mkdir();shutil.copy2(__file__,snap/'analyze.py');dump(OUT/'status.json',{'status':'completed_read_only_no_followups','no_forward_training_or_new_matching':True});print('ALL_COMPLETE',flush=True)
if __name__=='__main__':main()
