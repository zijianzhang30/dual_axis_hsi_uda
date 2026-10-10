import csv,hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,cohen_kappa_score

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_source_multi_prototype_diagnostic_v1'
CACHE=ROOT/'results/houston_routing_signal_diagnostic_v1';DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS=(2100,2101,2102);K=4
spec=importlib.util.spec_from_file_location('mp_utils',ROOT/'investigations/houston_target_boundary_signal_diagnostic_v1/run.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
sha,dump,aggregate=u.sha,u.dump,u.aggregate
def unit(x):
 n=np.linalg.norm(x,axis=-1,keepdims=True);assert (n>0).all();return x/n
def write(name,rows):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def metrics(y,p):
 cm=confusion_matrix(y,p,labels=np.arange(7));rec=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(7,float),where=cm.sum(1)>0)
 return {'OA':100*float((y==p).mean()),'AA':100*float(rec.mean()),'Kappa':float(cohen_kappa_score(y,p,labels=np.arange(7)))},cm

def main():
 OUT.mkdir(exist_ok=False);gate=json.loads((CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text());assert not gate['target_GT_opened_by_constructor']
 old=json.loads((CACHE/'provenance.json').read_text())
 def copy(rel):
  assert sha(CACHE/rel)==gate['files_sha256'][rel];dest=OUT/rel;dest.parent.mkdir(exist_ok=True);shutil.copyfile(CACHE/rel,dest);assert sha(dest)==gate['files_sha256'][rel]
 for name in ['source_labels.npy','source_train_mask.npy']:copy(name)
 sy=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy');assert train.sum()==2393 and (~train).sum()==127
 for seed in SEEDS:
  rec=old['seeds'][str(seed)];assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
  for name in ['source_Z.npy','target_Z.npy','target_logits.npy','target_predictions.npy','source_coords.npy','target_coords.npy','source_sample_indices.npy','target_sample_indices.npy','R1_predictions.npy','R1_centroid_scores.npz']:copy(f'seed_{seed}/{name}')
 dump(OUT/'provenance.json',{'historical_metadata':old['historical_metadata'],'seeds':old['seeds'],'sklearn_version':sklearn.__version__,
  'code_sha256':{str(p):sha(p) for p in HERE.iterdir() if p.is_file()},'utility_sha256':sha(Path(u.__file__)),'cache_manifest_sha256':sha(CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json')})
 inputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()};dump(OUT/'ALL_INPUTS_FROZEN.json',{'target_GT_opened':False,'files_sha256':inputs})
 source_rows=[];sizes=[];compactness=[]
 for seed in SEEDS:
  folder=OUT/f'seed_{seed}';source=unit(np.load(folder/'source_Z.npy').astype(float));target=unit(np.load(folder/'target_Z.npy').astype(float));cent=[];proto=[];native=[];members=np.full(len(sy),-1,int)
  for c in range(7):
   idx=np.flatnonzero(train&(sy==c));x=source[idx];mu=x.mean(0);cent.append(unit(mu))
   km=KMeans(n_clusters=K,init='k-means++',n_init=10,max_iter=300,tol=1e-4,algorithm='lloyd',random_state=0).fit(x)
   normalized=unit(km.cluster_centers_);proto.extend(normalized);native.extend(km.cluster_centers_);members[idx]=K*c+km.labels_
   sse1=float(((x-mu)**2).sum());cos=normalized@normalized.T
   compactness.append({'seed':seed,'class':f'C{c+1}','source_train_n':len(idx),'K1_SSE':sse1,'K4_SSE':float(km.inertia_),
    'K4_over_K1_SSE':float(km.inertia_/sse1),'prototype_offdiag_cosine_mean':float(cos[np.triu_indices(K,1)].mean()),
    'prototype_offdiag_cosine_min':float(cos[np.triu_indices(K,1)].min()),'Kmeans_n_iter':int(km.n_iter_)})
   for j in range(K):sizes.append({'seed':seed,'class':f'C{c+1}','prototype_in_class':j,'prototype_id':K*c+j,'source_train_n':int((km.labels_==j).sum())})
  cent=np.stack(cent);proto=np.stack(proto)
  np.savez_compressed(folder/'source_prototype_models.npz',class_centroids=cent,prototypes=proto,native_kmeans_centers=np.stack(native),prototype_classes=np.repeat(np.arange(7),K),source_train_membership=members)
  for domain,x in [('source',source),('target',target)]:
   cs=x@cent.T;ps=x@proto.T;pc=ps.reshape(len(x),7,K).max(2);npid=ps.argmax(1);pp=pc.argmax(1);cp=cs.argmax(1)
   assert np.array_equal(npid//K,pp)
   np.savez_compressed(folder/f'{domain}_frozen_scores.npz',centroid_scores=cs,prototype_scores=ps,prototype_class_scores=pc,
    centroid_predictions=cp,prototype_predictions=pp,nearest_prototype_ids=npid)
   if domain=='target':
    assert np.array_equal(cp,np.load(folder/'R1_predictions.npy'))
    previous=np.load(folder/'R1_centroid_scores.npz');assert np.array_equal(cent,previous['source_centroids']) and np.max(abs(cs-previous['target_scores']))<1e-12
   else:
    for split,mask in [('train',train),('val',~train)]:
     for name,p in [('centroid',cp),('prototype4',pp)]:source_rows.append({'seed':seed,'split':split,'router':name,**metrics(sy[mask],p[mask])[0]})
  print(seed,'source prototypes and all unlabeled target scores frozen',flush=True)
 write('source_prototype_sizes.csv',sizes);write('source_compactness_per_class.csv',compactness);write('source_router_accuracy.csv',source_rows)
 frozen={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()};dump(OUT/'ALL_PROTOTYPES_ROUTING_FROZEN.json',{'target_GT_opened':False,'all3_seeds_complete':True,'K_fixed4_perclass':True,'no_model_training_backprop_tuning':True,'files_sha256':frozen})
 for rel,h in frozen.items():assert sha(OUT/rel)==h
 print('All models/scores/predictions hash-frozen; now opening target GT for interpretation.',flush=True)
 y=np.load(DATA/'target_labels.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
 assert hashlib.sha256(y.tobytes()).hexdigest()==old['historical_metadata']['domains']['target']['labels_sha256']
 scoring(y,gt,source_rows)
 for rel,h in frozen.items():assert sha(OUT/rel)==h
 for rec in old['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
 dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_models_scores_checkpoints_unchanged':True,'target_GT_scoring_only_after_freeze':True,'no_auto_training':True})
 dump(OUT/'FINAL_DELIVERY.json',{'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FINAL_DELIVERY.json'}})

def scoring(y,gt,source_rows):
 summaries=[];subsets=[];categories=[];classrows=[];protorows=[];confrows=[]
 for seed in SEEDS:
  f=OUT/f'seed_{seed}';coords=np.load(f/'target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
  p=np.load(f/'target_predictions.npy');assert np.array_equal(p,np.load(f/'target_logits.npy').argmax(1));s=np.load(f/'target_frozen_scores.npz');cs=s['centroid_scores'];ps=s['prototype_class_scores'];cp=s['centroid_predictions'];pp=s['prototype_predictions'];err=p!=y
  for name,pred in [('R0',p),('centroid',cp),('prototype4',pp)]:
   mm,cm=metrics(y,pred);summaries.append({'seed':seed,'router':name,**mm,'R0_error_correction_percent':100*float((pred[err]==y[err]).mean()),'R0_correct_destruction_percent':100*float((pred[~err]!=y[~err]).mean())})
   for c in range(7):classrows.append({'seed':seed,'router':name,'class':f'C{c+1}','n':int(cm[c].sum()),'recall':100*cm[c,c]/cm[c].sum()})
   for a in range(7):
    for b in range(7):confrows.append({'seed':seed,'router':name,'true_class':f'C{a+1}','predicted_class':f'C{b+1}','n':int(cm[a,b])})
  masks={'all_target':np.ones(len(y),bool),'R0_correct':~err,'R0_errors':err}
  for c in range(7):masks[f'C{c+1}_all']=y==c;masks[f'C{c+1}_R0_errors']=err&(y==c)
  for a in range(7):
   for b in range(7):
    if a!=b:masks[f'C{a+1}_to_C{b+1}']=err&(y==a)&(p==b)
  for label,mask in masks.items():
   if not mask.any():continue
   rr=np.arange(len(y));cwrong=cs.copy();pwrong=ps.copy();cwrong[rr,y]=-np.inf;pwrong[rr,y]=-np.inf
   cmargin=cs[rr,y]-cwrong.max(1);pmargin=ps[rr,y]-pwrong.max(1)
   e=mask&err;ce=cp==y;pe=pp==y
   row={'seed':seed,'subset':label,'n':int(mask.sum()),'centroid_correct_percent':100*ce[mask].mean(),'prototype_correct_percent':100*pe[mask].mean(),
    'prototype_minus_centroid_accuracy_pp':100*(pe[mask].mean()-ce[mask].mean()),'centroid_true_vs_best_wrong_margin_mean':float(cmargin[mask].mean()),'prototype_true_vs_best_wrong_margin_mean':float(pmargin[mask].mean())}
   if e.any():
    row.update({'n_R0_errors':int(e.sum()),'centroid_agrees_R0_wrong_class_percent':100*(cp[e]==p[e]).mean(),'prototype_agrees_R0_wrong_class_percent':100*(pp[e]==p[e]).mean(),
     'prototype_R0_wrong_class_centroid_other_class_percent':100*((pp[e]==p[e])&(cp[e]!=p[e])).mean(),
     'centroid_true_correct_prototype_R0_wrong_class_percent':100*((cp[e]==y[e])&(pp[e]==p[e])).mean(),
     'prototype_R0_wrong_class_support_margin_positive_percent':100*(ps[rr[e],p[e]]>ps[rr[e],y[e]]).mean(),
     'centroid_R0_wrong_class_support_margin_positive_percent':100*(cs[rr[e],p[e]]>cs[rr[e],y[e]]).mean(),
     'wrong_class_score_gain_mean':float((ps[rr[e],p[e]]-cs[rr[e],p[e]]).mean()),'true_class_score_gain_mean':float((ps[rr[e],y[e]]-cs[rr[e],y[e]]).mean()),
     'wrong_vs_true_relative_attraction_gain_mean':float(((ps[rr[e],p[e]]-ps[rr[e],y[e]])-(cs[rr[e],p[e]]-cs[rr[e],y[e]])).mean())})
   subsets.append(row)
   for category,cm in [('both_correct',ce&pe),('centroid_only_correct',ce&~pe),('prototype_only_correct',~ce&pe),
    ('both_wrong_same_class',~ce&~pe&(cp==pp)),('both_wrong_different_class',~ce&~pe&(cp!=pp))]:
    categories.append({'seed':seed,'subset':label,'category':category,'n':int((mask&cm).sum()),'fraction_percent':100*cm[mask].mean()})
  for pid in range(28):
   mask=s['nearest_prototype_ids']==pid;n=int(mask.sum());counts=np.bincount(y[mask],minlength=7)
   protorows.append({'seed':seed,'prototype_id':pid,'source_class':f'C{pid//K+1}','target_n':n,'target_R0_error_n':int((mask&err).sum()),
    'source_class_target_purity_percent':100*counts[pid//K]/n if n else None,**{f'target_C{c+1}_n':int(counts[c]) for c in range(7)}})
 write('routing_metrics_per_seed.csv',summaries);write('per_class_recall.csv',classrows);write('router_confusions.csv',confrows)
 write('local_attraction_subsets_per_seed.csv',subsets);write('centroid_vs_prototype_categories.csv',categories);write('target_assignment_per_prototype.csv',protorows)
 sa=aggregate(summaries,['router'],['OA','AA','Kappa','R0_error_correction_percent','R0_correct_destruction_percent'])
 keys=[k for k in subsets[0] if k not in ['seed','subset','n']];extra=[k for r in subsets for k in r if k not in ['seed','subset','n'] and k not in keys];keys+=list(dict.fromkeys(extra))
 su=aggregate(subsets,['subset'],keys);ca=aggregate(categories,['subset','category'],['fraction_percent']);ra=aggregate(classrows,['router','class'],['recall'])
 write('routing_metrics_mean_std.csv',sa);write('local_attraction_subsets_mean_std.csv',su);write('categories_mean_std.csv',ca);write('recall_mean_std.csv',ra)
 out={'routing':sa,'subsets':su,'categories':ca,'source_accuracy':source_rows,'no_target_GT_in_prototypes_or_routing':True,'no_training_or_auto_followup':True};dump(OUT/'summary.json',out)
 report(out)

def report(out):
 lines=['Houston source multi-prototype frozen Z diagnostic; originalC3seeds epoch200; source train2393 only.',
  'Fixed4prototypes/class, ordinary KMeans with fixed seed0/n_init10; class score=max4cosines. No transport/main-model training/tuning.',
  'All source models, source/target scores and predictions hashed before target GT. Centroid predictions exactly replay priorR1.',
  'Mean+/-sampleSD across3modelseeds.','\nTarget routing metrics:']
 lines.extend(str(r) for r in out['routing']);lines.append('\nSource heldout:');lines.extend(str(r) for r in out['source_accuracy'] if r['split']=='val')
 lines.append('\nOriginal-error interpretation and focus subsets:')
 lines.extend(str(r) for r in out['subsets'] if r['subset'] in ['R0_errors','R0_correct','C1_R0_errors','C7_R0_errors','C7_to_C6','C5_to_C6','C4_to_C6'])
 lines.append('\nOriginal-error centroid/prototype categories:');lines.extend(str(r) for r in out['categories'] if r['subset']=='R0_errors')
 lines.extend(['\nInterpretive limits:',
  'KMeans always splits each class into4groups; source SSE improvement is not a multimodality test.',
  'Wrong-source-prototype attraction describes a failure, not proof that adding more prototypes can correct it.',
  'Compare prototype-only correct vs centroid-only correct, local-specific wrong-class attraction vs already global wrong attraction, and wrong-vs-true differential score gains.',
  'Prototypes use source only; unrepresented target modes/domain-shifted subclasses may remain closer to wrong source classes.',
  'No target GT selected K/stage/threshold/model/weight, no numerical Go threshold inferred from these scores, no auto new training.'])
 text='\n'.join(lines)+'\n';(OUT/'REPORT.txt').write_text(text);(OUT/'FINDINGS.txt').write_text(text)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
 routers=[r for r in out['routing'] if r['router']!='R0'];x=np.arange(2)
 ax[0].bar(x-.16,[r['R0_error_correction_percent_mean'] for r in routers],.32,yerr=[r['R0_error_correction_percent_std'] for r in routers],label='R0 errors corrected')
 ax[0].bar(x+.16,[r['R0_correct_destruction_percent_mean'] for r in routers],.32,yerr=[r['R0_correct_destruction_percent_std'] for r in routers],label='R0 correct destroyed')
 ax[0].set(xticks=x,xticklabels=[r['router'] for r in routers],ylabel='Subset fraction (%)');ax[0].legend()
 cats=[r for r in out['categories'] if r['subset']=='R0_errors'];ax[1].bar(np.arange(5),[r['fraction_percent_mean'] for r in cats],yerr=[r['fraction_percent_std'] for r in cats]);ax[1].set(xticks=np.arange(5),xticklabels=[r['category'].replace('_','\n') for r in cats],ylabel='R0 errors (%)');ax[1].tick_params(axis='x',labelsize=8)
 fig.savefig(OUT/'multi_prototype_error_geometry.png',dpi=180);plt.close(fig)
 print(text[:6000],flush=True)
if __name__=='__main__':main()
