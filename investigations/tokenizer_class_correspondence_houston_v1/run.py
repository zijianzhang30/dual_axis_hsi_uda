import sys,json,csv,time
from pathlib import Path
import numpy as np,torch
from common import OUT,BASE,CACHE,SEEDS,dump,sha,hasharray
SEM=BASE.parent/'tokenizer_semantic_diagnostics_v1'
HEAD=BASE.parent/'tokenizer_head_diagnostic_houston_v1'
ARMS=('A','C');STAGES=('T','Z','T_mean')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def descript(x):
 return {'n':int(len(x)),'mean':float(x.mean()),'std':float(x.std()),'median':float(np.median(x)),'q05':float(np.percentile(x,5)),'q95':float(np.percentile(x,95))} if len(x) else None
def metric(y,score):
 pred=score.argmax(1);cm=np.bincount(7*y+pred,minlength=49).reshape(7,7);count=cm.sum(1);recall=np.diag(cm)/count
 return {'OA':float(100*np.trace(cm)/len(y)),'AA':float(100*recall.mean()),'recall_percent':(100*recall).tolist(),'confusion':cm.tolist()}
def subset_geometry(mask,ts,zs):
 n=int(mask.sum())
 if not n:return {'n':0}
 t=ts[mask];z=zs[mask];tp=t.argmax(1);zp=z.argmax(1);tm=t[:,5]-t[:,4];zm=z[:,5]-z[:,4];trans=np.bincount(7*tp+zp,minlength=49).reshape(7,7)
 return {'n':n,'T_nearest_source_class_counts':np.bincount(tp,minlength=7).tolist(),'Z_nearest_source_class_counts':np.bincount(zp,minlength=7).tolist(),'nearest_source_T_to_Z_counts':trans.tolist(),'T_C6_minus_C5_cosine':descript(tm),'Z_C6_minus_C5_cosine':descript(zm),'T_C6_minus_C5_positive_fraction':float((tm>0).mean()),'Z_C6_minus_C5_positive_fraction':float((zm>0).mean()),'T_nearest_source_C5_fraction':float((tp==4).mean()),'T_nearest_source_C6_fraction':float((tp==5).mean()),'Z_nearest_source_C5_fraction':float((zp==4).mean()),'Z_nearest_source_C6_fraction':float((zp==5).mean()),'nearest_source_C6_to_C5_n':int(((tp==5)&(zp==4)).sum()),'nearest_source_C5_to_C6_n':int(((tp==4)&(zp==5)).sum()),'nearest_source_C5_to_C5_n':int(((tp==4)&(zp==4)).sum()),'pair_margin_C6_preferred_T_to_C5_preferred_Z_n':int(((tm>0)&(zm<0)).sum()),'pair_margin_C5_preferred_T_to_C6_preferred_Z_n':int(((tm<0)&(zm>0)).sum())}
def csvwrite(name,rows):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarise(v):
 a=np.asarray(v,float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1))}

def main():
 start=time.monotonic();torch.set_num_threads(2);sl=np.load(CACHE/'data/source_labels.npy');tl=np.load(CACHE/'data/target_labels.npy');coords=np.load(CACHE/'data/source_coords.npy');tcoords=np.load(CACHE/'data/target_coords.npy');assert np.array_equal(tl,np.load(BASE/'houston/evaluation_labels.npy'))
 runs={};files={};cp_hashes={};centroidrows=[];classrows=[];subsetrows=[];transitionrows=[];finalrows=[];sensitivityrows=[]
 for seed in SEEDS:
  folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=False);cached=SEM/f'seed_{seed}'
  assert np.array_equal(coords,np.load(cached/'source_coordinates.npy'));assert np.array_equal(tcoords,np.load(cached/'target_coordinates.npy'));assert np.array_equal(sl,np.load(cached/'source_labels_offline.npy'));assert np.array_equal(tl,np.load(cached/'target_labels_offline.npy'))
  train=np.load(HEAD/f'seed_{seed}/fixed_head_parameters.npz')['source_train_mask'];assert train.sum()==2393 and (~train).sum()==127
  srcconfig=json.loads((BASE/'houston'/f'C_{seed}/config.json').read_text());predictions={a:np.load(BASE/'houston'/f'{a}_{seed}/predictions_200.npz')['predictions'] for a in ARMS};c6=tl==5
  masks={'all_target_C6':c6,'C_final_C6_to_C5':c6&(predictions['C']==4),'A_final_C6_to_C5':c6&(predictions['A']==4),'both_final_C6_to_C5':c6&(predictions['A']==4)&(predictions['C']==4),'C_recovers_A_C6_to_C5':c6&(predictions['A']==4)&(predictions['C']==5),'C_introduces_C6_to_C5_from_A_correct':c6&(predictions['A']==5)&(predictions['C']==4)}
  np.savez_compressed(folder/'fixed_C6_masks_and_coordinates.npz',target_coordinates=tcoords,**masks);runs[str(seed)]={}
  for arm in ARMS:
   config=json.loads((BASE/'houston'/f'{arm}_{seed}/config.json').read_text());assert config['indices_sha256']==srcconfig['indices_sha256']
   cp=BASE/'houston'/f'{arm}_{seed}/checkpoints/epoch_200.pth';cph=sha(cp);cp_hashes[str(cp)]=cph;cachedmeta=json.loads((cached/f'{arm}_diagnostics.json').read_text());assert cachedmeta['checkpoint_sha256']==cph
   state=torch.load(cp,map_location='cpu',weights_only=True)['model'];features={d:{} for d in ['source','target']}
   for domain in features:
    for stage in ['T','Z']:
     path=(BASE/'houston'/f'{arm}_{seed}/tokens_200.npy') if domain=='target' and stage=='T' else cached/f'{arm}_{domain}_{stage}.npy'
     files[str(path)]=sha(path);v=np.load(path).astype(np.float64);assert len(v)==(len(sl) if domain=='source' else len(tl));features[domain][stage]=v.reshape(len(v),-1)
     if stage=='T':assert v.shape[1:]==(4,64);features[domain]['T_mean']=v.mean(1)
   # Authenticate cached Z against the official linear classifier outputs and target ordering.
   reconstructed=features['target']['Z']@state['nn1.weight'].numpy().T+state['nn1.bias'].numpy();official=np.load(BASE/'houston'/f'{arm}_{seed}/predictions_200.npz');assert np.array_equal(reconstructed.argmax(1),official['predictions']);assert abs(reconstructed-official['logits']).max()<2e-5
   record={'seed':seed,'arm':arm,'epoch':200,'checkpoint_sha256':cph,'native_prediction_reconstruction_exact':True,'source_train_counts':np.bincount(sl[train],minlength=7).tolist(),'stages':{},'C6_subsets':{},'source_centroid_sensitivity':{}};scorecache={}
   for stage in STAGES:
    s=features['source'][stage];t=features['target'][stage];centers=np.stack([s[train&(sl==c)].mean(0) for c in range(7)]);cent=unit(centers);tc=np.stack([t[tl==c].mean(0) for c in range(7)]);matrix=unit(tc)@cent.T;qs=unit(t)@cent.T;ss=unit(s)@cent.T;scorecache[stage]=qs
    np.savez_compressed(folder/f'{arm}_{stage}_source_centroid_scores.npz',source_centroids=centers,source_scores=ss,target_scores=qs,target_class_to_source_class_cosine=matrix)
    own=qs[np.arange(len(tl)),tl];other=qs.copy();other[np.arange(len(tl)),tl]=-np.inf;margin=own-other.max(1);cls=[]
    for c in range(7):
     m=tl==c;row={'seed':seed,'arm':arm,'stage':stage,'target_class':c+1,'target_count':int(m.sum()),'whole_class_centroid_nearest_source_class':int(matrix[c].argmax()+1),'matched_centroid_cosine':float(matrix[c,c]),'matched_centroid_margin':float(matrix[c,c]-np.delete(matrix[c],c).max()),'per_sample_correct_source_centroid_fraction':float((qs[m].argmax(1)==c).mean()),'mean_correct_vs_bestwrong_margin':float(margin[m].mean()),'negative_margin_fraction':float((margin[m]<0).mean()),'mean_cosine_to_source_C5':float(qs[m,4].mean()),'mean_cosine_to_source_C6':float(qs[m,5].mean())};classrows.append(row);cls.append(row)
     for j in range(7):centroidrows.append({'seed':seed,'arm':arm,'stage':stage,'target_class':c+1,'source_class':j+1,'centroid_cosine':float(matrix[c,j]),'centroid_cosine_distance':float(1-matrix[c,j]),'per_sample_mean_cosine':float(qs[m,j].mean()),'per_sample_mean_cosine_distance':float(1-qs[m,j].mean())})
    record['stages'][stage]={'descriptor_dimension':s.shape[1],'target_centroid_cosine_matrix_rows_target_columns_source':matrix.tolist(),'per_class':cls,'source_heldout_prototype_metrics':metric(sl[~train],ss[~train]),'target_prototype_metrics':metric(tl,qs)}
    uc=np.stack([unit(s[train&(sl==c)]).mean(0) for c in range(7)]);alternative=unit(t)@unit(uc).T;record['source_centroid_sensitivity'][stage]=alternative
   for kind,mask in {**masks,'own_final_C6_to_C5':c6&(predictions[arm]==4)}.items():
    d=subset_geometry(mask,scorecache['T'],scorecache['Z']);record['C6_subsets'][kind]=d
    if d['n']:
     subsetrows.append({'seed':seed,'arm':arm,'subset':kind,**{k:d[k] for k in ['n','T_nearest_source_C5_fraction','T_nearest_source_C6_fraction','Z_nearest_source_C5_fraction','Z_nearest_source_C6_fraction','T_C6_minus_C5_positive_fraction','Z_C6_minus_C5_positive_fraction','nearest_source_C6_to_C5_n','nearest_source_C5_to_C6_n','nearest_source_C5_to_C5_n','pair_margin_C6_preferred_T_to_C5_preferred_Z_n','pair_margin_C5_preferred_T_to_C6_preferred_Z_n']}})
     for i,row in enumerate(d['nearest_source_T_to_Z_counts']):
      for j,n in enumerate(row):transitionrows.append({'seed':seed,'arm':arm,'subset':kind,'T_nearest_source_class':i+1,'Z_nearest_source_class':j+1,'count':n,'fraction_of_subset':n/d['n']})
    finalrows.append({'seed':seed,'arm':arm,'subset':kind,'n':d['n']})
   for desc in ['T','T_mean']:
    for variant in ['raw_source_mean','unit_source_mean']:
     ts=scorecache[desc] if variant=='raw_source_mean' else record['source_centroid_sensitivity'][desc];zs=scorecache['Z'] if variant=='raw_source_mean' else record['source_centroid_sensitivity']['Z']
     for kind,mask in [('all_target_C6',c6),('own_final_C6_to_C5',c6&(predictions[arm]==4))]:
      d=subset_geometry(mask,ts,zs)
      if d['n']:sensitivityrows.append({'seed':seed,'arm':arm,'T_descriptor':desc,'centroid_method':variant,'subset':kind,'n':d['n'],'T_already_nearest_source_C5_fraction':d['T_nearest_source_C5_fraction'],'T_C6_preferred_over_C5_fraction':d['T_C6_minus_C5_positive_fraction'],'Z_nearest_source_C5_fraction':d['Z_nearest_source_C5_fraction'],'strict_nearest_source_C6_to_C5_fraction':d['nearest_source_C6_to_C5_n']/d['n'],'pair_C6_to_C5_sign_flip_fraction':d['pair_margin_C6_preferred_T_to_C5_preferred_Z_n']/d['n']})
   del record['source_centroid_sensitivity'];runs[str(seed)][arm]=record;dump(folder/f'{arm}_diagnostics.json',record);del features,scorecache,state
  print('completed',seed,flush=True)
 csvwrite('target_to_source_centroid_matrices.csv',centroidrows);csvwrite('per_class_correspondence.csv',classrows);csvwrite('C6_subset_stage_changes.csv',subsetrows);csvwrite('C6_T_to_Z_transition_counts.csv',transitionrows);csvwrite('C6_subgroup_counts.csv',finalrows);csvwrite('descriptor_centroid_sensitivity.csv',sensitivityrows)
 pooled={};aggregates={}
 for arm in ARMS:
  pooled[arm]={};aggregates[arm]={}
  for kind in ['all_target_C6','own_final_C6_to_C5','C_final_C6_to_C5']:
   rr=[runs[str(s)][arm]['C6_subsets'][kind] for s in SEEDS];matrix=np.sum([r['nearest_source_T_to_Z_counts'] for r in rr],axis=0);n=sum(r['n'] for r in rr)
   pooled[arm][kind]={'n_seed_pixel_observations':n,'not_independent_pixels':True,'nearest_source_T_to_Z_counts':matrix.tolist(),'T_already_nearest_source_C5_fraction':float(matrix[4].sum()/n),'T_nearest_source_C6_fraction':float(matrix[5].sum()/n),'Z_nearest_source_C5_fraction':float(matrix[:,4].sum()/n),'strict_source_C6_T_to_source_C5_Z_fraction':float(matrix[5,4]/n),'strict_source_C5_T_to_source_C6_Z_fraction':float(matrix[4,5]/n),'pair_C6_T_to_C5_Z_fraction':sum(r['pair_margin_C6_preferred_T_to_C5_preferred_Z_n'] for r in rr)/n}
   aggregates[arm][kind]={k:summarise([r[k] for r in rr]) for k in ['T_nearest_source_C5_fraction','T_nearest_source_C6_fraction','Z_nearest_source_C5_fraction','Z_nearest_source_C6_fraction','T_C6_minus_C5_positive_fraction','Z_C6_minus_C5_positive_fraction']}
 pairedclass=[]
 for seed in SEEDS:
  for stage in STAGES:
   a=runs[str(seed)]['A']['stages'][stage]['per_class'];c=runs[str(seed)]['C']['stages'][stage]['per_class']
   for ca,cc in zip(a,c):pairedclass.append({'seed':seed,'stage':stage,'target_class':ca['target_class'],'C_minus_A_correct_source_nearest_fraction':cc['per_sample_correct_source_centroid_fraction']-ca['per_sample_correct_source_centroid_fraction'],'C_minus_A_correct_vs_wrong_margin':cc['mean_correct_vs_bestwrong_margin']-ca['mean_correct_vs_bestwrong_margin'],'A_whole_class_nearest_source':ca['whole_class_centroid_nearest_source_class'],'C_whole_class_nearest_source':cc['whole_class_centroid_nearest_source_class']})
 csvwrite('paired_C_minus_A_class_correspondence.csv',pairedclass)
 for path,h in files.items():assert sha(path)==h
 for path,h in cp_hashes.items():assert sha(path)==h
 result={'seeds':SEEDS,'fixed_epoch':200,'runs':runs,'aggregate_C6_subsets':aggregates,'pooled_C6_subset_transitions':pooled,'no_training':True,'GT_only_offline_geometry_and_subsets':True,'source_centroids_source_train_only':True,'input_checkpoints_and_feature_caches_unchanged':True,'seconds':time.monotonic()-start,'limitations':['T flatten256 vs Z64 geometry scales differ: nearest-class/sign transitions are descriptive, not causal evidence Transformer introduced the classification error.','Each model own final-error subset differs; C_final_C6_to_C5 provides identical target pixel subset for paired A/C.','Final errors condition on outcome; all C6 harm and rescue transitions separately reported.','A/C have separately trained CNN/Transformer weights; cannot uniquely attribute differences to tokenizer assignment.','Source-centroid probes and class means do not prove general separability or exclude multimodal target classes.','Pooled counts repeat same target pixels across3 seeds and are not independent samples.','Target C4 only22pixels; full class-centroid and per-sample statistics both reported.']};dump(OUT/'summary.json',result);dump(OUT/'provenance.json',{'checkpoint_sha256':cp_hashes,'cache_file_sha256':files,'code_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'protocol_sha256':sha(OUT/'PROTOCOL.txt'),'Pavia_modified':False});print(json.dumps(pooled),flush=True)
if __name__=='__main__':main()
