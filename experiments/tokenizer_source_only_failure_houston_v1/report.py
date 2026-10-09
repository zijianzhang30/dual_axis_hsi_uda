import json,csv,itertools
import numpy as np
from common import OUT,SEEDS,ARMS,sha,dump

def summarize(v):return {'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)),'values':[float(x) for x in v]}
def main():
 root=OUT/'report';root.mkdir(exist_ok=False);trajectories={seed:json.loads((OUT/'diagnostics'/f'seed_{seed}'/'trajectory.json').read_text()) for seed in SEEDS};assert len({v['fixed_target_indices_sha256'] for v in trajectories.values()})==1
 rows=[r for t in trajectories.values() for r in t['rows']];runs={a:[] for a in ARMS};pairing=[]
 for seed in SEEDS:
  folders={a:OUT/'houston'/f'{a}_{seed}' for a in ARMS};histories={a:[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()] for a,f in folders.items()};assert all(len(h)==200 for h in histories.values())
  for a,b in zip(histories['A'],histories['C']):
   for key in ['batch_stream_sha256','rng_stream_sha256']:assert a[key]==b[key]
  pairing.append({'seed':seed,'all200_source_data_augmentation_CPU_CUDA_rng_paired':True})
  for arm in ARMS:
   m=json.loads((folders[arm]/'frozen_manifest.json').read_text());assert not m['target_data_loaded_or_forwarded'];assert m['data_access']==['Houston13']
   for cp in m['checkpoints']:assert sha(cp['path'])==cp['sha256']
   row=next(r for r in rows if r['seed']==seed and r['arm']==arm and r['epoch']==200);metric=row['official_epoch200_full_target_metrics'];runs[arm].append({'seed':seed,'metrics':metric,'source_train_OA':histories[arm][-1]['source_train_OA'],'source_val':m['source_val_at_final'],'train_seconds':m['train_seconds'],'parameters':m['config']['parameters'],'training_target_reads_or_forwards':0})
 aggregate={a:{k:summarize([r['metrics'][k] for r in runs[a]]) for k in ['oa_percent','aa_percent','kappa']} for a in ARMS};paired=[{'seed':a['seed'],**{k:c['metrics'][k]-a['metrics'][k] for k in ['oa_percent','aa_percent','kappa']}} for a,c in zip(runs['A'],runs['C'])];epochs=[0,1,10,50,100,150,200];functional=[];divergence=[]
 for arm in ARMS:
  for epoch in epochs:
   rr=[r for r in rows if r['arm']==arm and r['epoch']==epoch]
   for name in ['source','target']:
    for stage in ['V_mean','V_center','T','Z']:
     matrix=[np.asarray(r['domains'][name]['stages'][stage]['centroid_cosine_matrix']) for r in rr];dist=[float(np.sqrt(np.mean((a-b)**2))) for a,b in itertools.combinations(matrix,2)]
     divergence.append({'arm':arm,'epoch':epoch,'domain':name,'stage':stage,'mean_pairwise_seed_class_geometry_RMS':float(np.mean(dist)),'macro_margin':summarize([r['domains'][name]['stages'][stage]['macro_margin_LOO'] for r in rr]),'feature_norm':summarize([r['domains'][name]['stages'][stage]['norm_mean'] for r in rr])})
    for layer in ['bn3d','bn2d']:
     functional.append({'arm':arm,'epoch':epoch,'domain':name,'layer':layer,'normalized_mean_mismatch_RMS':summarize([r['domains'][name]['BN_functional_mismatch'][layer]['input_running_mean_z_rms'] for r in rr]),'median_input_running_variance_ratio':summarize([r['domains'][name]['BN_functional_mismatch'][layer]['input_running_var_ratio_quantiles'][2] for r in rr])})
 literal=[]
 for arm in ARMS:
  for seed in SEEDS:
   h=[json.loads(x) for x in (OUT/'houston'/f'{arm}_{seed}'/'history.jsonl').read_text().splitlines()]
   for layer in ['bn3d','bn2d']:literal.append({'arm':arm,'seed':seed,'layer':layer,'first_epoch_running_var_below_1e8':next((r['epoch'] for r in h if r['BN'][layer]['running_var_min']<1e-8),None),'no_nonfinite_training_stats':True})
 result={'protocol':'Houston true source-only training; ordinaryBN; single input;200epochs; target diagnostics after all6 runs frozen','historical_75_18_not_source_only':True,'runs':runs,'aggregate':aggregate,'paired_C_minus_A':paired,'pairing':pairing,'BN_functional_mismatch_trajectory':functional,'stage_cross_seed_divergence':divergence,'literal_BN_variance_failure_checks':literal,'target_trajectory_metrics_are_stratified_sample_estimates':True,'observational_not_causal':'BN discrepancy does not establish BN as the cause; channel identities are not aligned across trained seeds. No automatic normalization/module design.'};dump(root/'summary.json',result)
 with (root/'final_metrics.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','seed','OA','AA','Kappa','source_train_OA','source_val_OA','collapse','zero_recall_classes'])
  for arm,rr in runs.items():
   for r in rr:w.writerow([arm,r['seed'],*[r['metrics'][k] for k in ['oa_percent','aa_percent','kappa']],r['source_train_OA'],r['source_val']['oa_percent'],r['metrics']['collapse'],r['metrics']['zero_recall_classes']])
 with (root/'trajectory_per_class.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','seed','epoch','class','source_recall','target_sample_recall','target_sample_prediction_share','sample_collapse'])
  for r in rows:
   s=r['domains']['source']['metrics'];t=r['domains']['target']['metrics']
   for k in range(7):w.writerow([r['arm'],r['seed'],r['epoch'],k+1,s['per_class_percent'][k],t['per_class_percent'][k],t['prediction_shares'][k],t['collapse']])
 text=['Houston source-only failure diagnosis, source only inputs, ordinaryBN momentum.1, A/C seeds2100/2101/2102, fixed200.','Historical75.18+-10.55 was sequential source+target sharedBN, not source-only.','Official final mean/sample-std '+json.dumps(aggregate),'Paired C-A '+json.dumps(paired),'BN physical variance checks '+json.dumps(literal),'All six training runs froze before target loading/GT evaluation; source data/RNG paired.','BN running mean/var channel-wise discrepancies across seeds do not prove failure: independently learned channels can differ. Compare normalized input mismatch and rotation-invariant class geometry across V_mean,V_center,T,Z.','Trajectory target sample stratified max256/class, rare class22; priorweightedOAestimate only. Formal OA is full52901 at200.','No target-best, target adaptation/statistics recalibration, new module or automatic follow-up.','Final per-run detail '+json.dumps(runs)]
 (root/'RESULTS.txt').write_text('\n'.join(text)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for arm in ARMS:
  for ax,key,label in [(axes[0],'trajectory_priorweighted_OA_estimate','Target OA estimate'),(axes[1],'aa_percent','Target AA')]:
   vals=np.array([[next(r for r in rows if r['arm']==arm and r['seed']==seed and r['epoch']==e)['domains']['target']['metrics'][key] for e in epochs] for seed in SEEDS]);mu=vals.mean(0);sd=vals.std(0,ddof=1);ax.plot(epochs,mu,label=arm);ax.fill_between(epochs,mu-sd,mu+sd,alpha=.15);ax.set_ylabel(label)
  vals=np.array([[next(r for r in rows if r['arm']==arm and r['seed']==seed and r['epoch']==e)['domains']['target']['stages']['Z']['macro_margin_LOO'] for e in epochs] for seed in SEEDS]);axes[2].plot(epochs,vals.mean(0),label=arm);axes[2].fill_between(epochs,vals.mean(0)-vals.std(0,ddof=1),vals.mean(0)+vals.std(0,ddof=1),alpha=.15);axes[2].set_ylabel('Target Z cosine class margin')
 for ax in axes:ax.set_xlabel('Fixed diagnostic epoch');ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(root/'source_only_trajectory.png',dpi=180);fig.savefig(root/'source_only_trajectory.svg');plt.close(fig);print(json.dumps({'aggregate':aggregate,'paired':paired}),flush=True)
if __name__=='__main__':main()
