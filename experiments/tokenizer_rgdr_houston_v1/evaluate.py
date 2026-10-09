import json,csv
import torch,numpy as np
from argparse import Namespace
from common import OUT,BASE,build,environment,loaders,sha,dump

def selection_stats(path):
 a=np.load(path);ids=a['selected'];sim=a['similarity'];att=a['attention'];b,k,m=ids.shape;counts=np.bincount(ids.flatten(),minlength=28);entropy=-(att*np.log(np.maximum(att,1e-30))).sum(-1);div=np.array([len(np.unique(x//4)) for x in ids.reshape(-1,m)]);chosen=np.take_along_axis(sim,ids,axis=-1)
 return {'queries':b*k,'memory_entries_per_query':m,'selected_prototype_counts':counts.tolist(),'selected_class_counts':np.bincount((ids//4).flatten(),minlength=7).tolist(),'selected_slot_counts':np.bincount((ids%4).flatten(),minlength=4).tolist(),'per_query_slot_prototype_counts':[np.bincount(ids[:,k].flatten(),minlength=28).tolist() for k in range(4)],'selected_memory_class_diversity_mean':float(div.mean()),'attention_entropy_mean':float(entropy.mean()),'attention_entropy_normalized_mean':float(entropy.mean()/np.log(m)),'selected_similarity_quantiles':np.quantile(chosen,[0,.1,.5,.9,1]).tolist(),'all_similarity_quantiles':np.quantile(sim,[0,.1,.5,.9,1]).tolist(),'prototype_inclusion_fraction':(counts/(b*k)).tolist(),'most_selected_prototype_inclusion_fraction':float(counts.max()/(b*k)),'distinct_selected_sets':len(set(tuple(sorted(x)) for x in ids.reshape(-1,m)))}

def main():
 torch.set_num_threads(2);env=environment();root=OUT/'evaluation_e100';root.mkdir(exist_ok=False);runs={};histories={};manifests={}
 for arm in ['P0','P1']:
  f=OUT/'houston'/f'{arm}_2100_e100';m=json.loads((f/'frozen_manifest.json').read_text());manifests[arm]=m;histories[arm]=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(histories[arm])==100
  for item in [*m['checkpoints'],m['prediction'],m['selection']]:assert sha(item['path'])==item['sha256']
 assert manifests['P0']['config']['initial_full_model_sha256']==manifests['P1']['config']['initial_full_model_sha256'];assert manifests['P0']['config']['parameters']==manifests['P1']['config']['parameters']
 for a,b in zip(histories['P0'],histories['P1']):
  for k in ['batch_stream_sha256','original_rng_stream_sha256','memory_class_updates','memory_valid_classes']:assert a[k]==b[k]
 labels=np.load(BASE/'houston/evaluation_labels.npy')
 for arm,m in manifests.items():
  a=np.load(m['prediction']['path']);metric=env[6].metrics(labels,a['predictions'],a['logits'],7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[i+1 for i,v in enumerate(metric['per_class_percent']) if v==0]
  runs[arm]={'metrics':metric,'parameters':m['config']['parameters'],'memory_T_W_bytes':m['config']['memory_T_W_bytes'],'all_memory_buffers_bytes':m['config']['all_memory_buffers_bytes'],'train_seconds':m['train_seconds'],'source_val':m['source_val_at_final'],'selection_stats':selection_stats(m['selection']['path']),'diagnostic_trajectory':[{k:v for k,v in r.items() if k in ['epoch','unlabeled_target_memory_diagnostic']} for r in histories[arm] if 'unlabeled_target_memory_diagnostic' in r]}
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=4,seed=2100);env[4](2100);loader=loaders(args)[3];model,_=build(2100,'C','houston');cp=BASE/'houston/C_2100/checkpoints/epoch_100.pth';model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.cuda().eval();logits=[]
 with torch.inference_mode():
  for x,_ in loader:x=x.cuda();logits.append(model(x,x)[1].cpu().numpy())
 logits=np.concatenate(logits);np.savez_compressed(root/'C_2100_e100_predictions.npz',logits=logits,predictions=logits.argmax(1));metric=env[6].metrics(labels,logits.argmax(1),logits,7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[i+1 for i,v in enumerate(metric['per_class_percent']) if v==0]
 assert abs(metric['oa_percent']-82.60524375720686)<1e-9;history=[json.loads(x) for x in (BASE/'houston/C_2100/history.jsonl').read_text().splitlines()];runs['C']={'metrics':metric,'parameters':378011,'source_val':history[99]['source_val'],'checkpoint_sha256':sha(cp),'train_seconds':None}
 keys=['oa_percent','aa_percent','kappa'];paired={f'{a}-{b}':{k:runs[a]['metrics'][k]-runs[b]['metrics'][k] for k in keys} for a,b in [('P0','C'),('P1','C'),('P1','P0')]}
 def clear(d):return max(d['oa_percent'],d['aa_percent'])>=.5 and min(d['oa_percent'],d['aa_percent'])>=-.5
 go=clear(paired['P1-P0']) and clear(paired['P1-C']) and not runs['P1']['metrics']['collapse'] and not runs['P1']['metrics']['zero_recall_classes']
 result={'dataset':'Houston','seed':2100,'fixed_epoch100':True,'runs':runs,'paired_deltas':paired,'primary':'P1-P0,architecture/parameters/main_RNG/data/init matched;selection only differs','exploratory_go_signal':go,'predeclared_signal_threshold':'max OA/AA delta>=.5pp and min>=-.5pp for P1-P0 and P1-C, no collapse/zero recall worsening','target_best_used':False,'target_only_inference':True,'source_and_unlabeled_target_training_JointBN':True,'no_auto_followup':True,'wall_time_caveat':'Concurrent/shared GPU loads make walltime descriptive rather than a controlled overhead benchmark.'};dump(root/'summary.json',result)
 with (root/'per_class_recall.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['class','target_count','C','P0','P1','P1-C','P1-P0'])
  for k in range(7):
   c,p0,p1=[runs[a]['metrics']['per_class_percent'][k] for a in ['C','P0','P1']];w.writerow([k+1,int((labels==k).sum()),c,p0,p1,p1-c,p1-p0])
 text=['Houston RGDR exploratory,seed2100,fixed100;target52901nativepixels.','Paired deltas '+json.dumps(paired),'Exploratory Go signal '+str(go),'Projection parametersP0==P1;main data/augmentation/CPU/CUDA RNG exactvsC across100epochs.','Memory source-only updates, previous-step rawsourceT/W, detached, fixedEMA.9;Top-M8,residual.1.','Target-only saved-memory inference verified; training still uses target via FullJointBN.','No target-best, no auto continuation/sweep/newloss. Walltime affected by shared GPU load.']
 for arm,r in runs.items():text.append(arm+' '+json.dumps(r))
 (root/'RESULTS.txt').write_text('\n'.join(text)+'\n');print(json.dumps({'paired':paired,'go':go}),flush=True)
if __name__=='__main__':main()
