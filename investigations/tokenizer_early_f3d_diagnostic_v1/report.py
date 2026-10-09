import json,csv,numpy as np
from common import OUT,SEEDS,dump

def stat(v):return {'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)),'values':[float(x) for x in v]}
rows=[];results={seed:json.loads((OUT/f'seed_{seed}'/'summary.json').read_text()) for seed in SEEDS};aggregate={}
for name in results[SEEDS[0]]['probes_and_geometry']:
 rr=[]
 for seed,r in results.items():
  s=r['probes_and_geometry'][name];g=s['geometry'];rr.append({'seed':seed,'descriptor':name,'dimension':s['dimension'],'source_probe_OA':s['probe_source']['oa_percent'],'target_probe_OA':s['probe_target']['oa_percent'],'target_probe_AA':s['probe_target']['aa_percent'],'source_class_margin':g['source']['macro_margin_LOO'],'target_class_margin':g['target']['macro_margin_LOO'],'target_to_source_centroid_margin':g['source_to_target']['macro_target_to_source_center_margin'],'same_class_centroid_cosine':g['source_to_target']['macro_matched_center_cosine'],'converged':s['converged']})
 rows+=rr;aggregate[name]={k:stat([a[k] for a in rr]) for k in rr[0] if k not in ['seed','descriptor','dimension','converged']}
curve_rows=[]
for seed,r in results.items():
 for record in r['perturb_diagnostics']['records']:
  for stage,v in record['stages'].items():curve_rows.append({'seed':seed,'condition':record['condition'],'strength':record['strength'],'stage':stage,'input_local_relative_energy_median':record['input_relative_energy_local3x3_quantiles'][2],'feature_relative_L2_change_median':v['feature_relative_L2_change_quantiles'][2],'feature_cosine_change_mean':v['feature_cosine_change_mean'],'response_gain_median':v['feature_response_gain_over_local_input_change_quantiles'][2] if v['feature_response_gain_over_local_input_change_quantiles'] is not None else None,'probe_OA_priorweighted_estimate':v['truepriorweighted_probe_OA_estimate'],'probe_AA':v['probe_AA'],'target_class_margin':v['target_class_margin'],'zero_feature_fraction':v['zero_feature_fraction']})
folder=OUT/'report';folder.mkdir(exist_ok=False)
for filename,data in [('stage_and_component_probes.csv',rows),('perturbation_sensitivity.csv',curve_rows)]:
 with (folder/filename).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
indices=[r['perturb_diagnostics']['indices_sha256'] for r in results.values()];assert len(set(indices))==1
probe_affine=[]
for seed in SEEDS:
 a=np.load(OUT/f'seed_{seed}/conv3d_probe.npz')['target_predictions'];b=np.load(OUT/f'seed_{seed}/bn3d_probe.npz')['target_predictions'];probe_affine.append({'seed':seed,'Conv3d_BN3d_source_standardized_probe_prediction_differences':int((a!=b).sum())})
dump(folder/'summary.json',{'architecture':results[SEEDS[0]]['architecture'],'probe_aggregate':aggregate,'probe_per_seed':rows,'perturbation_per_seed':curve_rows,'affine_probe_equivalence':probe_affine,'kernel_decomposition':{str(seed):r['decomposition'] for seed,r in results.items()},'ReLU_gating':{str(seed):r['ReLU_center_gating'] for seed,r in results.items()},'checkpoint_states_unchanged':all(r['state_and_checkpoint_unchanged'] for r in results.values()),'perturbations_not_energy_equivalent_do_not_compare_raw_drop_as_causal':True,'new_modules_or_training_launched':False,'recommendation':'pending interpretation, no automatic encoder design'})
text=['Houston formal C earlyF3D read-only diagnosis, seeds2100/2101/2102, epoch200.','One Conv3d, spectrum48->48,8channels;BN3d/ReLU. Flatten8x48=384 invertible. Firstchannelcompression occursConv2d384->64.','Source-only fixed probes on actual nodes, inputcenter/neighbors and kernel contributions. No targetfit or encodertraining.','Aggregate '+json.dumps(aggregate),'Affine BN probe '+json.dumps(probe_affine),'Spatial/spectral kernel decompositions reconstructpreactivation<1e-6;bias countedonce.','Perturbations on fixedtarget1558 classstratified;probeOA is true-priorweightedestimate, not official fullsceneOA. Clean fitprobes frozen.','Spectralgain/geometry severities differ; input energy and responsegain reported. No automatic inference of weakestdimension from unmatchedrawaccuracydrop.','Original checkpoints/runningbuffers unchanged;Pavia untouched. No module designed.']
(folder/'RESULTS.txt').write_text('\n'.join(text)+'\n')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4));names=['conv3d','bn3d','F3D','conv2d','bn2d','V'];xs=np.arange(6)
for seed in SEEDS:
 rr=[next(a for a in rows if a['seed']==seed and a['descriptor']==n) for n in names];axes[0].plot(xs,[a['target_probe_OA'] for a in rr],'-o',label=str(seed))
axes[0].set_xticks(xs,['Conv3D','BN3D','ReLU3D','Conv2D','BN2D','ReLU2D'],rotation=35);axes[0].set_ylabel('Target probe OA (%)');axes[0].legend()
for mode in ['spectral_all','spectral_center','spectral_neighbors']:
 rr=[a for a in curve_rows if a['condition']==mode and a['stage']=='F3D'];x=[.02,.05,.10];y=[np.mean([a['probe_AA'] for a in rr if a['strength']==v]) for v in x];axes[1].plot(x,y,'-o',label=mode)
axes[1].set_xlabel('Smooth spectral gain strength');axes[1].set_ylabel('F3D target probe AA (%)');axes[1].legend(fontsize=8)
names=['spatial_center','spatial_neighbors','spectral_offset0','spectral_neighbors'];axes[2].bar(np.arange(4),[aggregate[n]['target_probe_OA']['mean'] for n in names]);axes[2].set_xticks(np.arange(4),['center spatial','neighbor spatial','same band','adjacent bands'],rotation=30);axes[2].set_ylabel('Component target probe OA (%)');fig.tight_layout();fig.savefig(folder/'early_f3d_diagnosis.png',dpi=180);fig.savefig(folder/'early_f3d_diagnosis.svg');plt.close(fig);print('EarlyF3D report complete',flush=True)
