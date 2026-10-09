import json,torch,numpy as np
from common import OUT,BASE,SEEDS,sha,environment,dump
from diagnostics import describe,PERMS

def stats(v):
 x=np.array(v,float);return {'values':x.tolist(),'mean':float(x.mean()),'sample_std':float(x.std(ddof=1))}
def main():
 torch.set_num_threads(2);root=OUT/'houston';env=environment();old=json.loads((BASE/'houston/summary.json').read_text());manifests={}
 for seed in SEEDS:
  f=root/f'H_{seed}';m=json.loads((f/'frozen_manifest.json').read_text());manifests[seed]=m
  h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];ref=[json.loads(x) for x in (BASE/'houston'/f'C_{seed}/history.jsonl').read_text().splitlines()]
  assert len(h)==200 and h[-1]['epoch']==200
  for a,b in zip(h,ref):
   for k in ['batch_stream_sha256','original_rng_stream_sha256']:assert a[k]==b[k]
  for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens'],*m['components'].values()]:assert sha(item['path'])==item['sha256']
 dump(root/'all_predictions_frozen.json',{'all_3_H_epoch200_frozen':True,'all200_pairing_passed':True,'target_best':False})
 labels=np.load(BASE/'houston/evaluation_labels.npy');runs={};slotdiag={}
 for seed,m in manifests.items():
  f=root/f'H_{seed}';a=np.load(m['prediction']['path']);metric=env[6].metrics(labels,a['predictions'],a['logits'],7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0]
  arrays=[np.load(f/(k+'_200.npy'),mmap_mode='r') for k in ['weights_sem','weights_rel','tokens_sem','tokens_rel']];parts=[];sizes=[]
  for start in range(0,len(labels),128):
   tensors=[torch.from_numpy(np.array(x[start:start+128])) for x in arrays];parts.append(describe(*tensors));sizes.append(len(tensors[0]))
  weights=np.array(sizes)/sum(sizes);d={}
  for k in ['pairwise_weight_cosine_matrix','pairwise_weight_JS_matrix']:
   d[k]=np.average(np.array([p[k] for p in parts]),axis=0,weights=weights).tolist()
  for k in ['nonidentity_best_fraction','diagonal_JS','best_aligned_JS','diagonal_cosine','T_sem_rel_cosine','T_sem_rel_matched_cosine']:
   d[k]=float(np.average([p[k] for p in parts],weights=weights))
  d['best_matching_histogram']=np.sum([p['best_matching_histogram'] for p in parts],axis=0).tolist()
  j=np.array(d['pairwise_weight_JS_matrix']);d['best_global_slot_matching']=list(min(PERMS,key=lambda p:j[np.arange(4),list(p)].mean()))
  d['serious_slot_permutation']=d['nonidentity_best_fraction']>=.75 and d['diagonal_JS']-d['best_aligned_JS']>=.01;slotdiag[str(seed)]=d
  history=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()]
  r={'arm':'H','seed':seed,'metrics':metric,'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],'inference_seconds':m['inference_seconds'],'final_g':m['final_g'],'final_gradient_mean_norm':history[-1]['gradient_mean_norm'],'slot_diagnostic':d,'source_val':m['source_val_at_final']};runs[f'H_{seed}']=r;dump(f/'results.json',r)
  for arm in 'AC':runs[f'{arm}_{seed}']=old['runs'][f'{arm}_{seed}']
 keys=['oa_percent','aa_percent','kappa'];aggregate={arm:{**{k:stats([runs[f'{arm}_{s}']['metrics'][k] for s in SEEDS]) for k in keys},'parameters':runs[f'{arm}_{SEEDS[0]}']['parameters'],'train_seconds':stats([runs[f'{arm}_{s}']['train_seconds'] for s in SEEDS])} for arm in ['A','C','H']}
 for arm in ['A','C','H']:
  aggregate[arm]['per_class_recall']={str(k+1):stats([runs[f'{arm}_{s}']['metrics']['per_class_percent'][k] for s in SEEDS]) for k in range(7)}
  aggregate[arm]['zero_recall_by_seed']={str(s):runs[f'{arm}_{s}']['metrics']['zero_recall_classes'] for s in SEEDS}
  aggregate[arm]['collapse_count']=sum(runs[f'{arm}_{s}']['metrics']['collapse'] for s in SEEDS)
 paired={f'H-{a}':{k:stats([runs[f'H_{s}']['metrics'][k]-runs[f'{a}_{s}']['metrics'][k] for s in SEEDS]) for k in keys} for a in 'AC'}
 z=paired['H-C'];majority=sum(z['oa_percent']['values'][i]>0 and z['aa_percent']['values'][i]>0 for i in range(3))>=2
 mean_median=all(z[k]['mean']>=.25 and np.median(z[k]['values'])>0 for k in ['oa_percent','aa_percent'])
 gate_ok=all(.1<runs[f'H_{s}']['final_g']<.9 for s in SEEDS)
 grads_ok=all(runs[f'H_{s}']['final_gradient_mean_norm'][k]>1e-8 for s in SEEDS for k in ['semantic','relation'])
 go={'GO':bool(majority and mean_median and gate_ok and grads_ok),'majority_same_seed_OA_AA_positive':bool(majority),'mean_ge_quarter_pp_and_positive_median':bool(mean_median),'all_gate_between_point1_point9':gate_ok,'both_branches_final_grad_mean_gt1e-8':grads_ok,'no_automatic_followup':True}
 result={'dataset':'houston','primary':'fixed_epoch200','aggregate':aggregate,'paired_deltas':paired,'runs':runs,'gate_by_seed':{str(s):runs[f'H_{s}']['final_g'] for s in SEEDS},'slot_diagnostics':slotdiag,'parameter_overhead_vs_C':257,'parameter_overhead_vs_A':1701,'compute_overhead_MACs_per_patch_vs_C':86528,'compute_overhead_MACs_per_patch_vs_A':324480,'gate_blend_arithmetic_ops_per_patch':512,'go_no_go':go,'Pavia_unchanged':True,'no_target_best':True}
 dump(root/'summary.json',result)
 lines=['Houston Semantic-Relational Dual-Token H; fixed200,seeds2100/2101/2102','Arm\tOA\tAA\tKappa\tParameters\tTrainSeconds']
 for a,z in aggregate.items():lines.append(a+'\t'+'\t'.join(f"{z[k]['mean']:.4f}+/-{z[k]['sample_std']:.4f}" for k in keys)+f"\t{z['parameters']}\t{z['train_seconds']['mean']:.2f}")
 for name,z in paired.items():lines.append(name+' '+json.dumps(z))
 for name,r in runs.items():lines.append(name+' '+json.dumps(r))
 lines.extend(['Go/No-Go '+json.dumps(go),'Parameter overhead vsC257,vsA1701; added tokenizer MACs86528/patch plus softmax and512 gate arithmetic ops. SameCNN/Transformer count.','Training timing overhead vsC '+str([runs[f'H_{s}']['train_seconds']/runs[f'C_{s}']['train_seconds']-1 for s in SEEDS]),'Timing confounded by concurrent GPU loads. Slot matching is diagnostic only; no alignment applied. No extra experiment.'])
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
