import sys,json,csv,time
from argparse import Namespace
from pathlib import Path
import numpy as np,torch
from common import OUT,BASE,CACHE,SEEDS,frozen,dump,sha,hasharray

HEADS=('linear','linear_no_bias','cosine_weight','source_centroid_cosine')
def norm(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def stats(x):
 x=np.asarray(x,float);return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'median':float(np.median(x)),'q05':float(np.percentile(x,5)),'q95':float(np.percentile(x,95))}
def metric(y,score):
 pred=score.argmax(1);cm=np.bincount(y*7+pred,minlength=49).reshape(7,7);counts=cm.sum(1);recall=np.diag(cm)/counts;prior=counts/len(y);pp=cm.sum(0)/len(y);oa=np.trace(cm)/len(y);chance=prior@pp
 def kl(p,q):return float(np.sum(p*np.log(np.maximum(p,1e-30)/np.maximum(q,1e-30))))
 mid=(prior+pp)/2
 return {'n':len(y),'OA':float(100*oa),'AA':float(100*recall.mean()),'Kappa':float((oa-chance)/(1-chance)),'recall_percent':(100*recall).tolist(),'confusion':cm.tolist(),'actual_prior':prior.tolist(),'prediction_prior':pp.tolist(),'prior_delta_pp':(100*(pp-prior)).tolist(),'prior_total_variation':float(abs(pp-prior).sum()/2),'prior_JS':.5*(kl(prior,mid)+kl(pp,mid)),'collapse_prediction_share_ge95percent':bool(pp.max()>=.95),'zero_recall_classes':[i+1 for i,r in enumerate(recall) if r==0]}
def margins(y,score):
 truth=score[np.arange(len(y)),y];other=score.copy();other[np.arange(len(y)),y]=-np.inf;othermax=other.max(1);margin=truth-othermax;pred=score.argmax(1);ordered=np.sort(score,axis=1);predmargin=ordered[:,-1]-ordered[:,-2]
 return {str(c+1):{'true_vs_bestother_margin':stats(margin[y==c]),'negative_margin_fraction':float((margin[y==c]<0).mean()),'mean_own_score':float(truth[y==c].mean()),'mean_strongest_other_score':float(othermax[y==c].mean()),'predicted_top1_top2_margin':stats(predmargin[y==c]),'main_rival_class_counts':np.bincount(other[y==c].argmax(1),minlength=7).tolist()} for c in range(7)}
def summary(values):
 v=np.asarray(values,float);return {'seeds':SEEDS,'values':v.tolist(),'mean':float(v.mean()),'sample_std':float(v.std(ddof=1))}
def writecsv(path,rows):
 with Path(path).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 torch.set_num_threads(2);start=time.monotonic();env=frozen.environment();sources=np.load(CACHE/'data/source_labels.npy');coords=np.load(CACHE/'data/source_coords.npy');fixed=json.loads((CACHE/'fixed_inputs.json').read_text());input_records=[];fitted={};cache_hashes={}
 # Build all fixed source-only heads before opening target labels.
 for seed in SEEDS:
  cp=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth';cp_sha=sha(cp);assert cp_sha==fixed['checkpoints'][str(seed)]['sha256'];old=json.loads((CACHE/f'seed_{seed}/summary.json').read_text());assert cp_sha==old['checkpoint_sha256']
  state=torch.load(cp,map_location='cpu',weights_only=True)['model'];weight=state['nn1.weight'].numpy().copy();bias=state['nn1.bias'].numpy().copy();assert weight.shape==(7,64)
  env[4](seed);args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=seed);ls=frozen.loaders(args);train_indices=ls[0].dataset.indices;ref=json.loads((BASE/'houston'/f'C_{seed}/config.json').read_text());assert hasharray(train_indices)==ref['indices_sha256']['source'];assert np.array_equal(ls[3].dataset.indices,np.load(CACHE/'data/target_coords.npy'))
  trainset=set(map(tuple,train_indices));train=np.array([tuple(x) in trainset for x in coords]);assert train.sum()==2393 and (~train).sum()==127
  z={}
  for domain in ['source','target']:
   f=CACHE/f'seed_{seed}/{domain}_Z.npy';cache_hashes[str(f)]=sha(f);z[domain]=np.load(f);assert z[domain].shape[1]==64
  centers=np.stack([z['source'][train&(sources==c)].astype(np.float64).mean(0) for c in range(7)]).astype(np.float32);scores={}
  for domain in ['source','target']:
   v=z[domain];raw=v@weight.T
   scores[domain]={'linear':raw+bias,'linear_no_bias':raw,'cosine_weight':norm(v)@norm(weight).T,'source_centroid_cosine':norm(v)@norm(centers).T}
   cached=np.load(CACHE/f'seed_{seed}/{domain}_logits.npy');err=float(np.max(abs(scores[domain]['linear']-cached)));assert err<2e-5,(seed,domain,err);assert np.array_equal(scores[domain]['linear'].argmax(1),cached.argmax(1))
   if domain=='target':
    official=np.load(BASE/'houston'/f'C_{seed}/predictions_200.npz');assert np.array_equal(scores[domain]['linear'].argmax(1),official['predictions'])
  folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=False);np.savez_compressed(folder/'fixed_head_parameters.npz',linear_weight=weight,linear_bias=bias,source_centroids=centers,source_train_mask=train)
  for domain in ['source','target']:np.savez_compressed(folder/f'{domain}_fixed_scores.npz',**scores[domain])
  input_records.append({'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':cp_sha,'cached_feature_checkpoint_exact':True,'original_source_train_index_hash_exact':True,'source_train_mask_sha256':hasharray(train),'source_train_counts':np.bincount(sources[train],minlength=7).tolist(),'source_val_counts':np.bincount(sources[~train],minlength=7).tolist(),'classifier_weights_sha256':hasharray(weight),'fixed_head_parameter_sha256':sha(folder/'fixed_head_parameters.npz'),'native_argmax_exact':True,'source_train_not_fixed2100_split':True})
  fitted[seed]=(z,weight,bias,centers,train,scores);del ls,state
 dump(OUT/'fixed_heads_before_target_scoring.json',{'heads':HEADS,'checkpoints':input_records,'target_fitting':False,'temperature_tuned':False,'all_heads_frozen_before_target_labels_loaded':True})
 target=np.load(CACHE/'data/target_labels.npy');assert np.array_equal(target,np.load(BASE/'houston/evaluation_labels.npy'));runs={};parameterrows=[];marginrows=[];priorrows=[];flowrows=[];pairedrows=[]
 for seed,(z,weight,bias,centers,train,scores) in fitted.items():
  rec={'seed':seed,'epoch':200,'source_train_counts':np.bincount(sources[train],minlength=7).tolist(),'source_heldout_counts':np.bincount(sources[~train],minlength=7).tolist(),'head_parameters':{'weight_norm':np.linalg.norm(weight,axis=1).tolist(),'bias':bias.tolist(),'weight_centroid_cosine':np.sum(norm(weight)*norm(centers),axis=1).tolist(),'weight_pairwise_cosine':(norm(weight)@norm(weight).T).tolist(),'source_centroid_pairwise_cosine':(norm(centers)@norm(centers).T).tolist()},'heads':{}}
  for c in range(7):parameterrows.append({'seed':seed,'class':c+1,'weight_L2_norm':float(np.linalg.norm(weight[c])),'bias':float(bias[c]),'weight_source_centroid_cosine':float(norm(weight)[c]@norm(centers)[c]),'source_train_count':int((sources[train]==c).sum())})
  original=scores['target']['linear'].argmax(1)
  for head in HEADS:
   d={'source_train':metric(sources[train],scores['source'][head][train]),'source_heldout':metric(sources[~train],scores['source'][head][~train]),'target':metric(target,scores['target'][head]),'margins':{}}
   for dom,lab,sco in [('source_train',sources[train],scores['source'][head][train]),('source_heldout',sources[~train],scores['source'][head][~train]),('target',target,scores['target'][head])]:
    d['margins'][dom]=margins(lab,sco)
    for c,r in d['margins'][dom].items():marginrows.append({'seed':seed,'head':head,'domain':dom,'class':int(c),'true_margin_mean':r['true_vs_bestother_margin']['mean'],'true_margin_median':r['true_vs_bestother_margin']['median'],'negative_margin_fraction':r['negative_margin_fraction'],'own_score_mean':r['mean_own_score'],'strongest_other_score_mean':r['mean_strongest_other_score']})
   pred=scores['target'][head].argmax(1);cm=np.asarray(d['target']['confusion']);c6wrong=(target==5)&(pred!=5)
   d['C6_wrong_destination_counts']=np.bincount(pred[c6wrong],minlength=7).tolist();d['C6_wrong_total']=int(c6wrong.sum());basewrong=(target==5)&(original!=5);basecorrect=(target==5)&(original==5)
   d['C6_recovered_original_errors']=int((basewrong&(pred==5)).sum());d['C6_lost_original_correct']=int((basecorrect&(pred!=5)).sum());d['per_class_changed_correct']=[int(((target==c)&(original!=c)&(pred==c)).sum()) for c in range(7)];d['per_class_changed_wrong']=[int(((target==c)&(original==c)&(pred!=c)).sum()) for c in range(7)]
   for c in range(7):
    priorrows.append({'seed':seed,'head':head,'class':c+1,'actual_target_prior_percent':100*d['target']['actual_prior'][c],'prediction_prior_percent':100*d['target']['prediction_prior'][c],'prediction_minus_actual_pp':d['target']['prior_delta_pp'][c],'target_recall_percent':d['target']['recall_percent'][c]})
    for j in range(7):flowrows.append({'seed':seed,'head':head,'true_class':c+1,'predicted_class':j+1,'count':int(cm[c,j]),'percent_of_true_class':100*float(cm[c,j]/cm[c].sum())})
   rec['heads'][head]=d
   if head!='linear':
    base=rec['heads']['linear']['target'];pairedrows.append({'seed':seed,'head':head,'delta_OA_pp':d['target']['OA']-base['OA'],'delta_AA_pp':d['target']['AA']-base['AA'],'delta_Kappa':d['target']['Kappa']-base['Kappa'],'delta_C6_recall_pp':d['target']['recall_percent'][5]-base['recall_percent'][5],'C6_recovered':d['C6_recovered_original_errors'],'C6_lost':d['C6_lost_original_correct']})
  wrong=(target==5)&(original!=5);p=original[wrong];zz=z['target'][wrong];feature_delta=np.einsum('nd,nd->n',zz,weight[5][None,:]-weight[p]);bias_delta=bias[5]-bias[p]
  rec['linear_C6_wrong_margin_decomposition']={'n':int(wrong.sum()),'Z_dot_weight_difference':stats(feature_delta),'bias_difference':stats(bias_delta),'total_C6_minus_predicted_logit':stats(feature_delta+bias_delta)}
  # Offline target centroids characterize geometry only; never passed to any classifier.
  unit=norm(z['target']);target_cent=np.stack([unit[target==c].mean(0) for c in range(7)]);target_cent=norm(target_cent);source_dir=norm(centers);corr=source_dir@target_cent.T
  rec['source_to_target_centroid_cosine_matrix_offline_GT_only']=corr.tolist();rec['correct_centroid_cosine_minus_closest_wrong_per_class']=[float(corr[c,c]-np.max(np.delete(corr[c],c))) for c in range(7)]
  runs[str(seed)]=rec;dump(OUT/f'seed_{seed}/diagnostics.json',rec)
 aggregate={h:{m:summary([runs[str(s)]['heads'][h]['target'][m] for s in SEEDS]) for m in ['OA','AA','Kappa']} for h in HEADS};paired={h:{m:summary([runs[str(s)]['heads'][h]['target'][m]-runs[str(s)]['heads']['linear']['target'][m] for s in SEEDS]) for m in ['OA','AA','Kappa']} for h in HEADS if h!='linear'}
 for h in paired:paired[h]['C6_recall_delta_pp']=summary([runs[str(s)]['heads'][h]['target']['recall_percent'][5]-runs[str(s)]['heads']['linear']['target']['recall_percent'][5] for s in SEEDS]);paired[h]['OA_AA_both_improve_seed_count']=sum(runs[str(s)]['heads'][h]['target']['OA']>runs[str(s)]['heads']['linear']['target']['OA'] and runs[str(s)]['heads'][h]['target']['AA']>runs[str(s)]['heads']['linear']['target']['AA'] for s in SEEDS)
 for r in input_records:assert sha(r['checkpoint'])==r['checkpoint_sha256']
 for name,expected in cache_hashes.items():assert sha(name)==expected
 for name,rows in [('classifier_parameters.csv',parameterrows),('per_class_margins.csv',marginrows),('target_priors_and_recall.csv',priorrows),('confusion_flows.csv',flowrows),('paired_head_changes.csv',pairedrows)]:writecsv(OUT/name,rows)
 result={'fixed_seeds':SEEDS,'epoch':200,'heads':HEADS,'aggregate_target':aggregate,'paired_vs_linear':paired,'runs':runs,'provenance':input_records,'checkpoint_and_Z_cache_unchanged':True,'seconds':time.monotonic()-start,'no_training':True,'target_labels_only_offline_scoring':True,'no_auto_head_innovation_or_experiment':True,'limitations':['Raw margin scales differ between logit and cosine heads; sign/error rates and within-head domain shift are meaningful comparisons.','Source centroids use each checkpoint own source training subset, not previous diagnostic fixed2100 probe partition.','Fixed heads differ in geometry, not only class prior; metric shifts alone cannot establish prior shift as cause.','Current features were trained with target-aware Full Joint BN; a source-only centroid head does not make model target-free.','Changing inference head is exploratory diagnostic, not a retrained controlled method result.']}
 dump(OUT/'summary.json',result);dump(OUT/'provenance.json',{'script_hashes':{p.name:sha(p) for p in OUT.glob('*.py')},'cached_Z_file_hashes':cache_hashes,'source_input_manifest':str(CACHE/'fixed_inputs.json'),'source_input_manifest_sha256':sha(CACHE/'fixed_inputs.json'),'inputs':input_records,'originals_unchanged':True,'Pavia_modified':False});print(json.dumps({'aggregate':aggregate,'paired':paired}),flush=True)
if __name__=='__main__':main()
