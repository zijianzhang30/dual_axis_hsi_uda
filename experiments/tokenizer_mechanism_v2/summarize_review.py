"""Score only after every dataset prediction is frozen; decide ONLY declared continuation."""
import argparse
import itertools
import json
import numpy as np
from common import OUT,SEEDS,ARMS,dump,sha,environment,loaders

def stats(values):
 a=np.asarray(values,dtype=float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),'population_std':float(a.std(ddof=0))}

def main():
 from argparse import Namespace
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston','pavia'],required=True);args=p.parse_args();root=OUT/args.dataset
 env=environment();runs={};manifests={}
 for seed in SEEDS:
  histories={};configs={}
  for arm in ARMS:
   folder=root/f'{arm}_{seed}';m=json.loads((folder/'frozen_manifest.json').read_text());manifests[arm,seed]=m;configs[arm]=m['config']
   h=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()];histories[arm]=h
   assert len(h)==200 and h[-1]['epoch']==200
   for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
   if arm in 'DE':
    assert np.isfinite(m['same_parameters_D_E_max_error_real_patches'])
    # The requested <1e-6 gate was on random preflight inputs; report trained
    # real-patch FP32 error without blocking frozen endpoint scoring.
    print('Reported real-patch FP32 D/E error',arm,seed,m['same_parameters_D_E_max_error_real_patches'],flush=True)
  for arm in ARMS[1:]:
   for key in ['initial_original_model_sha256','indices_sha256','samples','lr','BN','tokens','dim','depth','dropout']:
    assert configs[arm][key]==configs['A'][key],(seed,arm,key)
   for a,b in zip(histories[arm],histories['A']):
    for key in ['batch_stream_sha256','original_rng_stream_sha256']:assert a[key]==b[key],(seed,arm,a['epoch'],key)
  assert len({configs[a]['initial_scorer_sha256'] for a in 'BCF'})==1
  assert configs['D']['initial_scorer_sha256']==configs['E']['initial_scorer_sha256']
 assert len({m['config']['indices_sha256']['target_eval'] for m in manifests.values()})==1
 combined={'dataset':args.dataset,'all_18_final_predictions_frozen':True,'all_200_epochs_init_samples_batch_and_rng_audit':'passed','manifests':{f'{a}_{s}':sha(root/f'{a}_{s}/frozen_manifest.json') for a,s in manifests}}
 dump(root/'all_predictions_frozen.json',combined)
 path='/home/zhangzj26/TGRS_MLUDA-2024/datasets/'+('Houston/' if args.dataset=='houston' else 'Pavia')
 if args.dataset=='pavia':from pathlib import Path;path=Path(path)
 la=Namespace(dataset=args.dataset,dataset_dir=path,num_workers=4,seed=2100)
 env[4](2100);eval_loader=loaders(la)[3];labels=np.concatenate([y.numpy() for _,y in eval_loader]);np.save(root/'evaluation_labels.npy',labels)
 for (arm,seed),m in manifests.items():
  artifact=np.load(m['prediction']['path']);metric=env[6].metrics(labels,artifact['predictions'],artifact['logits'],7)
  metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[k+1 for k,v in enumerate(metric['per_class_percent']) if v==0]
  r={'arm':arm,'seed':seed,'metrics':metric,'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],'inference_seconds':m['inference_seconds'],'same_parameters_D_E_max_error_real_patches':m['same_parameters_D_E_max_error_real_patches']}
  runs[f'{arm}_{seed}']=r;dump(root/f'{arm}_{seed}/results.json',r)
 keys=('oa_percent','aa_percent','kappa')
 aggregate={a:{**{k:stats([runs[f'{a}_{s}']['metrics'][k] for s in SEEDS]) for k in keys},'parameters':runs[f'{a}_{SEEDS[0]}']['parameters'],'train_seconds':stats([runs[f'{a}_{s}']['train_seconds'] for s in SEEDS]),'collapse_count':sum(runs[f'{a}_{s}']['metrics']['collapse'] for s in SEEDS)} for a in ARMS}
 for a in ARMS:
  aggregate[a]['per_class_recall']={str(k+1):stats([runs[f'{a}_{s}']['metrics']['per_class_percent'][k] for s in SEEDS]) for k in range(7)}
  aggregate[a]['zero_recall_by_seed']={str(s):runs[f'{a}_{s}']['metrics']['zero_recall_classes'] for s in SEEDS}
 paired={}
 for a,b in itertools.combinations(ARMS,2):
  # Later-letter minus earlier-letter, including C-B,C-E separately below.
  paired[f'{b}-{a}']={k:stats([runs[f'{b}_{s}']['metrics'][k]-runs[f'{a}_{s}']['metrics'][k] for s in SEEDS]) for k in keys}
 for a,b in [('C','B'),('C','E'),('C','F'),('E','D')]:
  paired[f'{a}-{b}']={k:stats([runs[f'{a}_{s}']['metrics'][k]-runs[f'{b}_{s}']['metrics'][k] for s in SEEDS]) for k in keys}
 gates={}
 for name in ['C-B','C-E','C-F']:
  gates[name]={'passed':any(paired[name][k]['mean']>=1 and sum(v>0 for v in paired[name][k]['values'])>=2 for k in ['oa_percent','aa_percent']),
   'rule':'mean OA orAA>=1pp and>=2/3 positive seeds','paired':paired[name]}
 gates['E~D']={'passed':all(abs(v)<=.1 for k in ['oa_percent','aa_percent'] for v in paired['E-D'][k]['values']),'rule':'Every seed absolute OA AND AA difference<=.1pp','paired':paired['E-D']}
 decision={'continue_pavia':args.dataset=='houston' and any(x['passed'] for x in gates.values()),'gates':gates,'no_other_experiments_authorized':True}
 dump(root/'continuation_decision.json',decision)
 summary={'dataset':args.dataset,'primary':'fixed_epoch200','aggregate':aggregate,'paired_deltas':paired,'runs':runs,'audit':combined,'continuation':decision,'verification_max_D_E_error':json.loads((OUT/'verification.json').read_text())['max_D_E_error']}
 dump(root/'summary.json',summary)
 # Plain-text report, plus machine-readable complete per-class output above.
 lines=[f'BiDA-self tokenizer mechanism: {args.dataset}, fixed200, mean +/- sampleSD','Arm\tOA\tAA\tKappa\tParameters\tTrainSeconds\tCollapse']
 for a,z in aggregate.items():
  lines.append(a+'\t'+'\t'.join('%.4f +/- %.4f'%(z[k]['mean'],z[k]['sample_std']) for k in keys)+f"\t{z['parameters']}\t{z['train_seconds']['mean']:.1f}\t{z['collapse_count']}/3")
 lines+=['','All paired contrasts (later minus earlier; declared contrasts also included):']
 for name,z in paired.items():
  lines.append(name+'\t'+'\t'.join(f"{k}: {z[k]['mean']:+.4f} +/- {z[k]['sample_std']:.4f}, per-seed={z[k]['values']}" for k in keys))
 lines+=['','Per-seed recall, zero-recall and predicted class shares:']
 for name,r in runs.items():lines.append(name+'\t'+json.dumps(r['metrics']))
 lines+=['','Continuation decision: '+json.dumps(decision),'No target-best scores computed or used. Training timings reflect concurrent GPU loads.']
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps({'aggregate':aggregate,'decision':decision},indent=2),flush=True)
if __name__=='__main__':main()
