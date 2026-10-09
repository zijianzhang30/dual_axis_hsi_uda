"""Independent reports; existing A–F are read-only controls, G fixed200 only."""
import argparse
import json
import numpy as np
from common import OUT,BASE,SEEDS,environment,dump,sha

def stats(values):
 a=np.array(values,dtype=float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),'population_std':float(a.std(ddof=0))}

def summarize_AF():
 summaries={ds:json.loads((BASE/ds/'summary.json').read_text()) for ds in ['houston','pavia']}
 record={'source_summary_sha256':{ds:sha(BASE/ds/'summary.json') for ds in summaries},
  'datasets':{ds:{'aggregate':x['aggregate'],'paired_deltas':{n:x['paired_deltas'][n] for n in ['C-A','C-B','C-F','C-E']},'support_gates':x['continuation']['gates']} for ds,x in summaries.items()},
  'no_existing_result_modified':True,'D_E_theory':'For shared linear parameters, mean/center references differ only by a spatial constant canceled by softmax. Separate training metrics are trajectory differences, not linear-reference benefits.'}
 dump(OUT/'A_F_summary.json',record)
 lines=['Existing A-F first-stage summary (read-only, before any G training)','']
 for ds,x in summaries.items():
  lines.append(ds+' fixed200, mean +/- sampleSD')
  for a,z in x['aggregate'].items():lines.append(a+': '+', '.join(f"{k} {z[k]['mean']:.4f}+/-{z[k]['sample_std']:.4f}" for k in ['oa_percent','aa_percent','kappa']))
  for n in ['C-A','C-B','C-F','C-E']:
   lines.append(n+': '+', '.join(f"{k} {z['mean']:+.4f}, per-seed{z['values']}" for k,z in x['paired_deltas'][n].items()))
 lines+=['','No old results/checkpoints changed. D/E performance differences do not imply a linear center-reference gain.']
 (OUT/'A_F_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
 return summaries['pavia']

def stage(dataset):
 root=OUT/dataset;baseline=json.loads((BASE/dataset/'summary.json').read_text());helper=environment()[6];runs={}
 for seed in SEEDS:
  folder=root/f'G_{seed}';m=json.loads((folder/'frozen_manifest.json').read_text());c=json.loads((BASE/dataset/f'C_{seed}/frozen_manifest.json').read_text())
  for x in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(x['path'])==x['sha256']
  gh=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()];ch=[json.loads(x) for x in (BASE/dataset/f'C_{seed}/history.jsonl').read_text().splitlines()]
  assert len(gh)==len(ch)==200
  for g,old in zip(gh,ch):
   for key in ['epoch','steps','batch_stream_sha256','original_rng_stream_sha256']:assert g[key]==old[key],(seed,g['epoch'],key)
  for key in ['initial_original_model_sha256','initial_full_model_sha256','initial_scorer_sha256','parameters','indices_sha256','samples','lr','BN','tokens','dim','depth','dropout']:
   assert m['config'][key]==c['config'][key],(seed,key)
  runs[seed]={'manifest':m,'C_manifest_sha256':sha(BASE/dataset/f'C_{seed}/frozen_manifest.json')}
 dump(root/'all_G_predictions_frozen.json',{'passed':True,'original_A_F_unchanged':True,'same_C_G_init_capacity_samples_200epochs_rng':True,
  'G_manifests':{str(s):sha(root/f'G_{s}/frozen_manifest.json') for s in SEEDS}})
 labels=np.load(BASE/dataset/'evaluation_labels.npy');rows={}
 for seed,r in runs.items():
  m=r['manifest'];p=np.load(m['prediction']['path']);metric=helper.metrics(labels,p['predictions'],p['logits'],7)
  metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0]
  result={'arm':'G','seed':seed,'metrics':metric,'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],'inference_seconds':m['inference_seconds']}
  dump(root/f'G_{seed}/results.json',result);rows[f'G_{seed}']=result
 for a in 'ABCF':
  for s in SEEDS:rows[f'{a}_{s}']=baseline['runs'][f'{a}_{s}']
 aggregate={}
 for a in 'ABCFG':
  rs=[rows[f'{a}_{s}'] for s in SEEDS]
  aggregate[a]={k:stats([r['metrics'][k] for r in rs]) for k in ['oa_percent','aa_percent','kappa']}
  aggregate[a].update(parameters=rs[0]['parameters'],train_seconds=stats([r['train_seconds'] for r in rs]),collapse_count=sum(r['metrics']['collapse'] for r in rs),
   per_class_recall={str(k+1):stats([r['metrics']['per_class_percent'][k] for r in rs]) for k in range(7)},zero_recall_by_seed={str(s):rows[f'{a}_{s}']['metrics']['zero_recall_classes'] for s in SEEDS})
 paired={}
 for b in 'ABFG':paired[f'C-{b}']={k:stats([rows[f'C_{s}']['metrics'][k]-rows[f'{b}_{s}']['metrics'][k] for s in SEEDS]) for k in ['oa_percent','aa_percent','kappa']}
 for b in 'ABF':
  for k,z in paired[f'C-{b}'].items():assert z==baseline['paired_deltas'][f'C-{b}'][k]
 cg=paired['C-G'];consistent=sum(oa>0 and aa>0 for oa,aa in zip(cg['oa_percent']['values'],cg['aa_percent']['values']))
 passed=cg['oa_percent']['mean']>0 and cg['aa_percent']['mean']>0 and consistent>=2
 decision={'continue_Pavia_G':dataset=='houston' and passed,'C_G_support':passed,'paired_positive_OA_and_AA_seeds':consistent,'rule':'mean OA AND AA positive, >=2/3 seeds BOTH positive','no_other_experiments':True}
 scorer_count=1700 if dataset=='houston' else 3428
 summary={'dataset':dataset,'primary':'fixed200','aggregate':aggregate,'paired_deltas':paired,'runs':rows,'continuation':decision,
  'C_G_capacity_proof':{'C_scorer_parameters':scorer_count,'G_scorer_parameters':scorer_count,'all_total_parameters_equal':all(rows[f'C_{s}']['parameters']==rows[f'G_{s}']['parameters'] for s in SEEDS),'all_initial_tensors_and_every_epoch_streams_equal':True},
  'original_A_F_summary_sha256':sha(BASE/dataset/'summary.json'),'no_target_best':True}
 dump(root/'summary.json',summary)
 lines=[f'{dataset} A/B/C/F/G fixed200, mean +/- sampleSD','Arm\tOA\tAA\tKappa\tParams\tTrainingSeconds\tCollapse']
 for a,z in aggregate.items():lines.append(a+'\t'+'\t'.join(f"{z[k]['mean']:.4f}+/-{z[k]['sample_std']:.4f}" for k in ['oa_percent','aa_percent','kappa'])+f"\t{z['parameters']}\t{z['train_seconds']['mean']:.2f}\t{z['collapse_count']}/3")
 lines+=['','Paired contrasts:']
 for n,z in paired.items():lines.append(n+': '+', '.join(f"{k} {v['mean']:+.4f}+/-{v['sample_std']:.4f}, per-seed={v['values']}" for k,v in z.items()))
 lines+=['','Per-seed metrics/recall/confusion/shares/zero recall:']
 for n,r in rows.items():lines.append(n+': '+json.dumps(r['metrics']))
 lines+=['','Capacity proof: '+json.dumps(summary['C_G_capacity_proof']),'Continuation: '+json.dumps(decision),
  'Training time includes concurrent GPU load, not a speed benchmark. Existing A-F results reused unchanged, with frozen final control hashes. No target-best.']
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps({'aggregate':aggregate,'decision':decision}),flush=True)
 return decision

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston','pavia']);p.add_argument('--AF',action='store_true');args=p.parse_args()
 if args.AF:summarize_AF()
 else:stage(args.dataset)
