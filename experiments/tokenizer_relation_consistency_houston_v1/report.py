import json,numpy as np,torch
from argparse import Namespace
from common import OUT,BASE,SEEDS,sha,environment,build,loaders,dump
from diagnostics import describe

def stats(v):
 a=np.array(v,float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1))}
def main():
 torch.set_num_threads(2);env=environment();arms=['C1'];root=OUT/'houston'
 if all((root/f'C2_{s}/frozen_manifest.json').exists() for s in SEEDS):arms.append('C2')
 manifests={}
 for arm in arms:
  for seed in SEEDS:
   f=root/f'{arm}_{seed}';m=json.loads((f/'frozen_manifest.json').read_text());h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];ref=[json.loads(x) for x in (BASE/'houston'/f'C_{seed}/history.jsonl').read_text().splitlines()]
   assert len(h)==200 and h[-1]['epoch']==200
   for a,b in zip(h,ref):
    for k in ['batch_stream_sha256','original_rng_stream_sha256']:assert a[k]==b[k]
   for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
   manifests[arm,seed]=m
 dump(root/'all_requested_predictions_frozen.json',{'C1_complete':True,'C2_tested':'C2' in arms,'epochs':200,'artifacts_verified':True,'no_target_best':True})
 labels=np.load(BASE/'houston/evaluation_labels.npy');old=json.loads((BASE/'houston/summary.json').read_text());runs={};diagnostics={}
 # Deterministic common final evaluation probe, same patches and noise for all models.
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=2100)
 env[4](2100);dataset=loaders(args)[3].dataset
 indices=np.random.default_rng(72000).choice(len(dataset),4096,replace=False)
 patch=torch.stack([dataset[int(i)][0] for i in indices]);del dataset
 np.save(root/'diagnostic_indices.npy',indices)
 for arm in ['C0']+arms:
  for seed in SEEDS:
   if arm=='C0':
    r=old['runs'][f'C_{seed}'];r={**r,'arm':'C0'};checkpoint=BASE/'houston'/f'C_{seed}/checkpoints/epoch_200.pth'
   else:
    m=manifests[arm,seed];a=np.load(m['prediction']['path']);metric=env[6].metrics(labels,a['predictions'],a['logits'],7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0]
    r={'arm':arm,'seed':seed,'metrics':metric,'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],'source_val':m['source_val_at_final']};checkpoint=root/f'{arm}_{seed}/checkpoints/epoch_200.pth'
   model,_=build(seed,'C','houston');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
   pieces=[describe(model.relation_scores,part,seed=77700+i) for i,part in enumerate(patch.split(128))]
   z={k:float(np.mean([p[k] for p in pieces])) for k in pieces[0] if k!='relation_collapse'};z['relation_collapse']=z['uniform_fraction']>=.95 or z['identical_tokens_fraction']>=.95;diagnostics[f'{arm}_{seed}']=z;r['relation_diagnostics']=z
   runs[f'{arm}_{seed}']=r
   if arm!='C0':dump(root/f'{arm}_{seed}/results.json',r)
 keys=['oa_percent','aa_percent','kappa'];agg={a:{**{k:stats([runs[f'{a}_{s}']['metrics'][k] for s in SEEDS]) for k in keys},'parameters':runs[f'{a}_{SEEDS[0]}']['parameters'],'train_seconds':stats([runs[f'{a}_{s}']['train_seconds'] for s in SEEDS])} for a in ['C0']+arms}
 paired={}
 for a,b in [('C1','C0'),('C2','C0'),('C2','C1')]:
  if a not in arms:continue
  paired[a+'-'+b]={k:stats([runs[f'{a}_{s}']['metrics'][k]-runs[f'{b}_{s}']['metrics'][k] for s in SEEDS]) for k in keys}
 go={}
 for arm in arms:
  z=paired[arm+'-C0'];paired_positive=sum(z['oa_percent']['values'][i]>0 and z['aa_percent']['values'][i]>0 for i in range(3))>=2
  positive_mean=all(z[k]['mean']>0 and np.median(z[k]['values'])>0 for k in ['oa_percent','aa_percent'])
  relation_better=sum(diagnostics[f'{arm}_{s}']['RI_KL']<=.9*diagnostics[f'C0_{s}']['RI_KL'] for s in SEEDS)>=2
  no_collapse=all(not diagnostics[f'{arm}_{s}']['relation_collapse'] for s in SEEDS)
  source_ok=True
  for s in SEEDS:
   baseline=[json.loads(x) for x in (BASE/'houston'/f'C_{s}/history.jsonl').read_text().splitlines()][-1]
   endpoint=[json.loads(x) for x in (root/f'{arm}_{s}/history.jsonl').read_text().splitlines()][-1]
   source_ok &= endpoint['source_val']['oa_percent']>=baseline['source_val']['oa_percent']-1 and endpoint['source_ce']<=max(.02,2*baseline['source_ce'])
  go[arm]={'GO':bool(paired_positive and positive_mean and relation_better and no_collapse and source_ok),'majority_same_seed_OA_AA_positive':paired_positive,'mean_and_median_positive':bool(positive_mean),'RI_KL_at_least10percent_lower_2of3':relation_better,'no_relation_collapse':no_collapse,'source_recognition_and_CE_ok':bool(source_ok)}
 result={'dataset':'houston','primary':'fixed200','runs':runs,'aggregate':agg,'paired_deltas':paired,'diagnostics':diagnostics,'go_no_go':go,'parameter_overhead':0,'C2_status':'tested' if 'C2' in arms else 'not run: intrinsic equivariance, nonzero-RE preflight requirement failed','Pavia_automatic_launch':False,'no_auto_sweep_or_new_modules':True}
 dump(root/'summary.json',result)
 lines=['Houston target relation consistency, fixed200, paired seeds2100/2101/2102','C2: '+result['C2_status'],'Arm\tOA\tAA\tKappa\tParameters\tTrainSeconds']
 for a,z in agg.items():lines.append(a+'\t'+'\t'.join(f"{z[k]['mean']:.4f}+/-{z[k]['sample_std']:.4f}" for k in keys)+f"\t{z['parameters']}\t{z['train_seconds']['mean']:.2f}")
 for name,z in paired.items():lines.append(name+' '+json.dumps(z))
 for name,r in runs.items():lines.append(name+' '+json.dumps(r))
 for arm in arms:lines.append(arm+' Go/No-Go '+json.dumps(go[arm]));lines.append('Train time overhead ratios '+str([runs[f'{arm}_{s}']['train_seconds']/runs[f'C0_{s}']['train_seconds']-1 for s in SEEDS]))
 lines.append('Wall-clock overhead confounded by concurrent GPU jobs; not a controlled hardware benchmark. No automatic continuation.')
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
