import json,csv
import numpy as np
from common import OUT,BASE,SEEDS,environment,sha,dump

def main():
 root=OUT/'evaluation_e200';root.mkdir(exist_ok=False);helper=environment()[6];labels=np.load(BASE/'houston/evaluation_labels.npy');runs={a:[] for a in ['C','CQ1']};pairing=[]
 for seed in SEEDS:
  current=OUT/'houston'/f'CQ1_{seed}_e200';reference=BASE/'houston'/f'C_{seed}';cm=json.loads((current/'frozen_manifest.json').read_text());bm=json.loads((reference/'frozen_manifest.json').read_text())
  ch=[json.loads(l) for l in (current/'history.jsonl').read_text().splitlines()];bh=[json.loads(l) for l in (reference/'history.jsonl').read_text().splitlines()];assert len(ch)==len(bh)==200
  for x,y in zip(ch,bh):
   for key in ['batch_stream_sha256','original_rng_stream_sha256']:assert x[key]==y[key],(seed,x['epoch'],key)
  cc,bc=cm['config'],bm['config']
  for key in ['initial_original_model_sha256','initial_scorer_sha256','indices_sha256','samples','patch','batch','lr','depth','dropout','loss','BN','tokens','dim']:assert cc[key]==bc[key],(seed,key)
  for m in [cm,bm]:
   assert m['checkpoints'][-1]['epoch']==200
   for item in [m['checkpoints'][-1],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
  pairing.append({'seed':seed,'all200_data_augmentation_CPU_CUDA_streams_exact':True,'C_reused_after_checks':True,'initial_original_and_scorer_hashes_exact':True})
  for arm,m,h in [('C',bm,bh),('CQ1',cm,ch)]:
   data=np.load(m['prediction']['path']);metric=helper.metrics(labels,data['predictions'],data['logits'],7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0]
   runs[arm].append({'seed':seed,'metrics':metric,'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],'source_val':m['source_val_at_final'],'projection_diagnostics':{k:v for k,v in h[-1].items() if k.startswith('center_projection')},'checkpoint_sha256':m['checkpoints'][-1]['sha256']})
 keys=['oa_percent','aa_percent','kappa'];paired=[]
 for c,q in zip(runs['C'],runs['CQ1']):
  d={k:q['metrics'][k]-c['metrics'][k] for k in keys};paired.append({'seed':c['seed'],**d,'OA_AA_both_positive':d['oa_percent']>0 and d['aa_percent']>0,'recall_delta_pp':(np.array(q['metrics']['per_class_percent'])-np.array(c['metrics']['per_class_percent'])).tolist()})
 aggregate={a:{k:{'mean':float(np.mean([r['metrics'][k] for r in rr])),'std':float(np.std([r['metrics'][k] for r in rr],ddof=1))} for k in keys} for a,rr in runs.items()}
 delta_stats={k:{'mean':float(np.mean([r[k] for r in paired])),'std':float(np.std([r[k] for r in paired],ddof=1))} for k in keys}
 count=sum(r['OA_AA_both_positive'] for r in paired);go=count>=2
 result={'dataset':'Houston','seeds':list(SEEDS),'fixed_epoch':200,'sample_std_ddof':1,'runs':runs,'aggregate':aggregate,'paired_CQ1_minus_C':paired,'paired_delta_mean_std':delta_stats,'pairing_audit':pairing,'majority_OA_and_AA_same_seed_positive':count,'target_free_next_stage_authorized_by_user_condition':go,'target_free_launched':False,'target_best_used':False,'causal_caveat':'Positive Center-CLS benefit does not alone isolate reasoning from center-feature residual bypass.','wall_time_caveat':'Historical C and CQ1 use different shared GPU loads; elapsed time ratio is descriptive, not a controlled speed benchmark.'}
 dump(root/'summary.json',result);dump(OUT/'continuation_decision.json',{'formal_complete':True,'gate':'same-seed OA>0 and AA>0 in at least2 of3 seeds','positive_seed_count':count,'target_free_condition_passed':go,'target_free_launched':False,'no_other_variants_authorized':True})
 with (root/'per_seed_metrics.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','seed','OA','AA','Kappa','parameters','train_seconds','zero_recall_classes','collapse'])
  for arm,rr in runs.items():
   for r in rr:w.writerow([arm,r['seed'],*[r['metrics'][k] for k in keys],r['parameters'],r['train_seconds'],r['metrics']['zero_recall_classes'],r['metrics']['collapse']])
 with (root/'per_class_recall.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','class','target_count','C_recall','CQ1_recall','delta_pp'])
  for c,q in zip(runs['C'],runs['CQ1']):
   for k,(a,b) in enumerate(zip(c['metrics']['per_class_percent'],q['metrics']['per_class_percent'])):w.writerow([c['seed'],k+1,int((labels==k).sum()),a,b,b-a])
 lines=['Houston Center-CLS formal; seeds2100/2101/2102; fixed epoch200; sample std(ddof=1).','Aggregate '+json.dumps(aggregate),'Paired per seed '+json.dumps(paired),'Paired mean/std '+json.dumps(delta_stats),'Positive OA AND AA on same seed: '+str(count)+'/3','Target-free continuation condition: '+('PASS' if go else 'NO-GO'),'All original initial tensors, indices,200-epoch data/augmentation/CPU/CUDA RNG streams match official C.','CQ1 parameters382107 vs C378011 (+4096). All three Transformer blocks and losses/JointBN unchanged.','Center feature bypass remains a mechanism caveat. No target-best or Pavia modification.','Wall-time comparisons are descriptive due to different shared GPU loads.']
 for a,rr in runs.items():
  for r in rr:lines.append(a+' '+json.dumps(r))
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps({'paired':paired,'go':go}),flush=True)
if __name__=='__main__':main()
