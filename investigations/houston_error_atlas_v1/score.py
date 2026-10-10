import csv,json
from collections import Counter
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from common import *

def spatial(gt,y,coords):
 known=(gt>=0)&(gt<7);pad=np.pad(gt,1,constant_values=-99);a=np.zeros_like(known);b=a.copy();ann=a.copy()
 for dr in [-1,0,1]:
  for dc in [-1,0,1]:
   if not (dr or dc):continue
   n=pad[1+dr:1+dr+gt.shape[0],1+dc:1+dc+gt.shape[1]]
   a|=known&(n>=-1)&(n!=gt)
   b|=known&(n>=0)&(n<7)&(n!=gt)
   ann|=known&(n==-1)
 r,c=coords.T;maskdist=np.full(len(y),np.nan);semdist=maskdist.copy();info=[]
 for cls in range(7):
  mask=(gt==cls)&a;sem=(gt==cls)&b;sel=y==cls
  if mask.any():maskdist[sel]=distance_transform_edt(~mask)[r[sel],c[sel]]
  if sem.any():semdist[sel]=distance_transform_edt(~sem)[r[sel],c[sel]]
  info.append({'class':f'C{cls+1}','class_mask_boundary_pixels':int(mask.sum()),'observed_semantic_boundary_pixels':int(sem.sum()),'eval_n':int(sel.sum())})
 anndist=distance_transform_edt(~ann)[r,c] if ann.any() else np.full(len(y),np.nan)
 edge=np.minimum.reduce([r,c,gt.shape[0]-1-r,gt.shape[1]-1-c])
 return {'class_mask':maskdist,'semantic':semdist,'annotation':anndist,'scene_edge':edge},info

def fixed_groups(y,p,patterns):
 err=p!=y;masks={'all_target':np.ones(len(y),bool),'R0_correct':~err,'R0_errors':err}
 for c in range(7):
  masks[f'C{c+1}_all']=y==c;masks[f'C{c+1}_correct']=(y==c)&~err;masks[f'C{c+1}_errors']=(y==c)&err
 for a in range(7):
  for b in range(7):
   if a!=b:masks[f'C{a+1}_to_C{b+1}']=err&(y==a)&(p==b)
 for bits in PATTERNS:masks['trajectory_'+bits]=patterns==bits
 return masks

def main():
 gate=json.loads((OUT/'ALL_ATLAS_FEATURES_READOUTS_FROZEN.json').read_text());assert not gate['target_GT_opened'] and gate['all3_seeds_complete']
 for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
 prov=json.loads((OUT/'provenance.json').read_text())
 for path,h in prov['code_sha256'].items():assert sha(path)==h
 print('All features/raw/relation/readout hashes verified; opening target GT offline.',flush=True)
 y=np.load(DATA/'target_labels.npy');assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
 padded=np.load(DATA/'target_gt.npy');coords=np.load(OUT/'raw/target_original_coords.npy');pc=coords+6;assert np.array_equal(y,padded[pc[:,0],pc[:,1]])
 gt=padded[6:-6,6:-6];assert list(gt.shape)==prov['historical_metadata']['domains']['target']['shape']
 distances,boundinfo=spatial(gt,y,coords);write('GT_boundary_availability.csv',boundinfo)
 np.savez_compressed(OUT/'GT_spatial_distances.npz',labels=y,original_coords=coords,**distances)
 pairrows=[];unordered=[];coverage=[];boundrows=[];bclass=[];bdis=[];trajrows=[];flagsrows=[];marginrows=[];evidence=[];unions=[];emodes=[];jointrows=[];readoutrows=[];pooledjoint=Counter();total_errors=0
 spatial_maps=[]
 for seed in SEEDS:
  folder=OUT/f'seed_{seed}';p=np.load(folder/'target_predictions.npy');logits=np.load(folder/'target_logits.npy');rr=np.arange(len(y));err=p!=y;total_errors+=int(err.sum());nerr=int(err.sum())
  assert np.array_equal(pc,np.load(folder/'target_coords.npy'))
  f=np.load(folder/'ALL_FIXED_READOUTS.npz');scores={d:f[d+'_scores'] for d in DESCRIPTORS};pred={d:f[d+'_predictions'] for d in DESCRIPTORS};correct={d:pred[d]==y for d in DESCRIPTORS}
  patterns=np.array([''.join('1' if correct[d][i] else '0' for d in STAGES) for i in rr])
  # Five requested flags overlap; the8bit trajectory is the strict partition.
  v,t,z=[correct[d] for d in STAGES]
  flags={'early-hard':~v,'T-recovered':~v&t,'Z-recovered':~t&z,'T-to-Z-damaged':t&~z,'persistent-error':~v&~t&~z,
   'V-to-T-damaged':v&~t,'Z-centroid-correct_classifier-wrong':z&err}
  for flag,mask in flags.items():flagsrows.append({'seed':seed,'flag':flag,'all_target_n':int(mask.sum()),'all_target_percent':100*mask.mean(),'R0_error_n':int((mask&err).sum()),'R0_errors_percent':100*(mask&err).sum()/nerr})
  for bits,label in PATTERNS.items():
   mask=patterns==bits;trajrows.append({'seed':seed,'bits_VTZ':bits,'trajectory':label,'n':int(mask.sum()),'all_target_percent':100*mask.mean(),'R0_error_n':int((mask&err).sum()),'R0_errors_percent':100*(mask&err).sum()/nerr,'R0_error_rate_percent':100*err[mask].mean() if mask.any() else None})
  counts=np.bincount(y[err]*7+p[err],minlength=49).reshape(7,7);direct=[];und=[]
  for a in range(7):
   for b in range(7):
    if a==b:continue
    direct.append({'seed':seed,'pair':f'C{a+1}->C{b+1}','true_class':f'C{a+1}','predicted_class':f'C{b+1}','error_n':int(counts[a,b]),'all_errors_percent':100*counts[a,b]/nerr,'true_class_error_rate_percent':100*counts[a,b]/(y==a).sum()})
  direct.sort(key=lambda r:(-r['error_n'],r['pair']))
  for i,r in enumerate(direct):r['rank']=i+1
  pairrows.extend(direct)
  for a in range(7):
   for b in range(a+1,7):
    n=int(counts[a,b]+counts[b,a]);und.append({'seed':seed,'pair':f'C{a+1}/C{b+1}','error_n':n,'all_errors_percent':100*n/nerr})
  und.sort(key=lambda r:(-r['error_n'],r['pair']))
  for i,r in enumerate(und):r['rank']=i+1
  unordered.extend(und)
  for name,rs in [('directed_confusion',direct),('unordered_confusion',und)]:coverage.append({'seed':seed,'mode_definition':name,'top5_n':sum(r['error_n'] for r in rs[:5]),'total_errors':nerr,'top5_all_errors_percent':100*sum(r['error_n'] for r in rs[:5])/nerr,'top5_modes':';'.join(r['pair'] for r in rs[:5])})
  band=np.where(np.isfinite(distances['class_mask']),np.where(distances['class_mask']<=6,'boundary','interior'),'unobserved')
  modekeys=[(f'C{y[i]+1}->C{p[i]+1}',str(patterns[i]),str(band[i])) for i in np.flatnonzero(err)];joint=Counter(modekeys);pooledjoint.update(joint)
  ranked=sorted(joint.items(),key=lambda kv:(-kv[1],kv[0]));top5=sum(n for _,n in ranked[:5])
  coverage.append({'seed':seed,'mode_definition':'joint_confusion_trajectory_GTmask_band','top5_n':top5,'total_errors':nerr,'top5_all_errors_percent':100*top5/nerr,'top5_modes':';'.join('|'.join(k) for k,_ in ranked[:5])})
  for rank,(key,n) in enumerate(ranked,1):jointrows.append({'seed':seed,'rank':rank,'confusion':key[0],'bits_VTZ':key[1],'trajectory':PATTERNS[key[1]],'GTmask_band':key[2],'error_n':n,'all_errors_percent':100*n/nerr})
  # Spatial distributions and fixed boundary/interior splits.
  scopes={'all_target':np.ones(len(y),bool),**{f'C{c+1}':y==c for c in range(7)}}
  for dtype,dist in distances.items():
   if dtype=='scene_edge':continue
   finite=np.isfinite(dist);safe=(distances['annotation']>6)&(distances['scene_edge']>=6)
   for scope,sm in scopes.items():
    for control,cm in [('all',np.ones(len(y),bool)),('annotation_far_scene_safe',safe)]:
     for label,bm in [('boundary_le6',finite&(dist<=6)),('interior_gt6',finite&(dist>6)),('unobserved',~finite)]:
      mask=sm&cm&bm;n=int(mask.sum());row={'seed':seed,'distance_definition':dtype,'scope':scope,'control':control,'band':label,'n':n,'errors':int((mask&err).sum()),'error_rate_percent':100*err[mask].mean() if n else None,'all_error_share_percent':100*(mask&err).sum()/nerr}
      (boundrows if scope=='all_target' else bclass).append(row)
   for subset,sm in [('R0_correct',~err),('R0_errors',err)]:bdis.append({'seed':seed,'distance_definition':dtype,'subset':subset,**stats(dist[sm]),'unobserved_n':int((sm&~finite).sum())})
   bins=[('0',finite&(dist==0)),('(0,1]',finite&(dist>0)&(dist<=1)),('(1,3]',finite&(dist>1)&(dist<=3)),('(3,6]',finite&(dist>3)&(dist<=6)),('(6,12]',finite&(dist>6)&(dist<=12)),('>12',finite&(dist>12)),('unobserved',~finite)]
   for label,bm in bins:
    n=int(bm.sum());boundrows.append({'seed':seed,'distance_definition':dtype,'scope':'all_target','control':'fixed_distance_bins','band':label,'n':n,'errors':int((bm&err).sum()),'error_rate_percent':100*err[bm].mean() if n else None,'all_error_share_percent':100*(bm&err).sum()/nerr})
  masks=fixed_groups(y,p,patterns);masks['Z_centroid_errors']=~z;masks['R0_and_Z_centroid_errors']=err&~z
  for label,mask in flags.items():masks['flag_'+label]=mask
  margins={};own_dist={};wrong_dist={};strong={}
  for d in DESCRIPTORS:
   score=scores[d];wrong=score.copy();wrong[rr,y]=-np.inf;own=score[rr,y];other=wrong.max(1)
   margins[d]=own-other;own_dist[d]=np.sqrt(np.maximum(0,2-2*np.clip(own,-1,1)));wrong_dist[d]=np.sqrt(np.maximum(0,2-2*np.clip(other,-1,1)));strong[d]=margins[d]>0
   cm=confusion_matrix(y,pred[d],labels=np.arange(7));rec=100*np.diag(cm)/cm.sum(1)
   readoutrows.append({'seed':seed,'descriptor':d,'OA':100*correct[d].mean(),'AA':float(rec.mean()),'Kappa':float(cohen_kappa_score(y,pred[d]))})
  logwrong=logits.copy();logwrong[rr,y]=-np.inf;native_margin=logits[rr,y]-logwrong.max(1)
  union=np.logical_or.reduce([strong[d] for d in DESCRIPTORS]);evidencebits=np.array([''.join('1' if strong[d][i] else '0' for d in DESCRIPTORS) for i in rr])
  for subset,mask in masks.items():
   if not mask.any():continue
   for d in DESCRIPTORS:
    for metric,values in [('source_same_class_centroid_distance',own_dist[d]),('source_nearest_wrong_centroid_distance',wrong_dist[d]),('source_cosine_correct_class_margin',margins[d])]:marginrows.append({'seed':seed,'subset':subset,'descriptor':d,'metric':metric,**stats(values[mask])})
    evidence.append({'seed':seed,'subset':subset,'descriptor':d,'n':int(mask.sum()),'positive_true_class_evidence_n':int((mask&strong[d]).sum()),'positive_evidence_percent':100*strong[d][mask].mean(),'centroid_prediction_correct_percent':100*correct[d][mask].mean(),
     'positive_evidence_unique_among5_n':int((mask&strong[d]&~np.logical_or.reduce([strong[other] for other in DESCRIPTORS if other!=d])).sum())})
   marginrows.append({'seed':seed,'subset':subset,'descriptor':'native_classifier','metric':'true_vs_strongest_logit_margin',**stats(native_margin[mask])})
   unions.append({'seed':seed,'subset':subset,'n':int(mask.sum()),'any_of5_evidence_percent':100*union[mask].mean(),
    'raw_spectrum_correct_V_wrong_percent':100*(strong['S_RAW']&~correct['V'])[mask].mean(),'relation_correct_T_wrong_percent':100*(strong['R_FULL']&~correct['T'])[mask].mean(),
    'V_correct_T_wrong_percent':100*(v&~t)[mask].mean(),'T_correct_Z_wrong_percent':100*(t&~z)[mask].mean(),'Z_centroid_correct_classifier_wrong_percent':100*(z&err)[mask].mean()})
  for bits,n in sorted(Counter(evidencebits[err]).items(),key=lambda kv:(-kv[1],kv[0])):emodes.append({'seed':seed,'evidence_bits_VTZ_spectrum_relation':bits,'error_n':n,'all_errors_percent':100*n/nerr})
  # Explicit per-pixel export; all target GT-derived arrays are post-freeze artifacts.
  arrays={'sample_id':np.load(folder/'target_sample_indices.npy'),'row':coords[:,0],'col':coords[:,1],'GT_class_C1to7':y+1,'R0_prediction_C1to7':p+1,'R0_error':err,
   'VTZ_correctness_bits':patterns,'GTmask_band':band,'evidence_bits_VTZ_spectrum_relation':evidencebits,'native_true_vs_competitor_margin':native_margin,
   **{f'distance_{k}':v for k,v in distances.items()},**{f'flag_{k}':v for k,v in flags.items()}}
  for d in DESCRIPTORS:arrays.update({f'{d}_prediction_C1to7':pred[d]+1,f'{d}_source_same_class_distance':own_dist[d],f'{d}_source_wrong_class_distance':wrong_dist[d],f'{d}_source_margin':margins[d],f'{d}_positive_evidence':strong[d]})
  np.savez_compressed(folder/'OFFLINE_PIXEL_ATLAS.npz',**arrays)
  with (folder/'per_pixel_atlas.csv').open('x',newline='') as stream:
   writer=csv.writer(stream);writer.writerow(list(arrays));writer.writerows(zip(*(arrays[k] for k in arrays)))
  spatial_maps.append({'seed':seed,'predictions':p,'errors':err,'top5_pairs':[r['pair'] for r in direct[:5]],'patterns':patterns})
  print(seed,'pixel atlas/confusions/spatial/margins/evidence scored',flush=True)
 write('confusion_pair_contributions_per_seed.csv',pairrows);write('unordered_confusion_contributions_per_seed.csv',unordered);write('top5_error_coverage_per_seed.csv',coverage)
 write('joint_error_modes_per_seed.csv',jointrows);write('boundary_error_rates_per_seed.csv',boundrows);write('boundary_error_rates_per_class.csv',bclass);write('spatial_distance_distributions.csv',bdis)
 write('VTZ_exclusive_trajectories.csv',trajrows);write('VTZ_overlapping_mechanism_flags.csv',flagsrows);write('margin_distance_distributions.csv',marginrows)
 write('correct_class_evidence_per_seed.csv',evidence);write('evidence_union_and_stage_disagreements.csv',unions);write('evidence_bitpatterns_on_errors.csv',emodes);write('fixed_readout_metrics.csv',readoutrows)
 pooled=[]
 for rank,(key,n) in enumerate(sorted(pooledjoint.items(),key=lambda kv:(-kv[1],kv[0])),1):pooled.append({'rank':rank,'confusion':key[0],'bits_VTZ':key[1],'trajectory':PATTERNS[key[1]],'GTmask_band':key[2],'seed_pixel_error_observations':n,'fraction_all_error_observations_percent':100*n/total_errors})
 write('joint_modes_pooled_seed_pixel_observations.csv',pooled)
 ca=aggregate(coverage,['mode_definition'],['top5_all_errors_percent']);pa=aggregate(pairrows,['pair'],['error_n','all_errors_percent']);ba=aggregate(boundrows,['distance_definition','scope','control','band'],['error_rate_percent','all_error_share_percent']);ta=aggregate(trajrows,['bits_VTZ','trajectory'],['all_target_percent','R0_errors_percent','R0_error_rate_percent'])
 ea=aggregate(evidence,['subset','descriptor'],['positive_evidence_percent','centroid_prediction_correct_percent','positive_evidence_unique_among5_n']);ua=aggregate(unions,['subset'],[k for k in unions[0] if k not in ['seed','subset','n']]);ma=aggregate(readoutrows,['descriptor'],['OA','AA','Kappa'])
 write('top5_error_coverage_mean_std.csv',ca);write('confusion_contributions_mean_std.csv',pa);write('boundary_rates_mean_std.csv',ba);write('trajectories_mean_std.csv',ta);write('evidence_mean_std.csv',ea);write('evidence_union_mean_std.csv',ua);write('readout_metrics_mean_std.csv',ma)
 output={'coverage':ca,'confusions':pa,'boundary_rates':ba,'trajectories':ta,'evidence':ea,'unions':ua,'readouts':ma,'pooled_joint_top5':pooled[:5],
  'pooled_joint_top5_coverage_percent':sum(r['fraction_all_error_observations_percent'] for r in pooled[:5]),'total_seed_pixel_error_observations':total_errors,
  'target_GT_used_only_after_freeze':True,'no_training_tuning_auto_followup':True}
 dump(OUT/'summary.json',output)
 from report import report
 report(output,spatial_maps,gt,coords)
 for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
 for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
 dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_features_raw_relation_scores_checkpoint_hashes_unchanged':True,'no_training_tuning_followup':True})
 dump(OUT/'FINAL_DELIVERY.json',{'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FINAL_DELIVERY.json'}})
 print('ERROR ATLAS COMPLETE',flush=True)
if __name__=='__main__':main()
