import json,csv
import numpy as np
from common import OUT,SEEDS,dump

def stats(v):
 a=[x for x in v if x is not None];return {'mean':float(np.mean(a)) if a else None,'std':float(np.std(a,ddof=1)) if len(a)>1 else None,'valid_seeds':len(a),'values':v}
def main():
 folder=OUT/'report';folder.mkdir(exist_ok=False);results={seed:json.loads((OUT/f'seed_{seed}'/'summary.json').read_text()) for seed in SEEDS};stages={};rows=[]
 for stage in ['F3D','V','T','Z','F3D_mean','V_mean']:
  rr=[]
  for seed,r in results.items():
   s=r['stages'][stage];rr.append({'seed':seed,'stage':stage,'source_probe_OA':s['probe_source_heldout']['oa_percent'],'source_probe_AA':s['probe_source_heldout']['aa_percent'],'target_probe_OA':s['probe_target']['oa_percent'],'target_probe_AA':s['probe_target']['aa_percent'],'source_class_margin':s['geometry']['source']['macro_margin_LOO'],'target_class_margin':s['geometry']['target']['macro_margin_LOO'],'target_to_source_class_margin':s['source_probe_train_centroid_target_geometry']['macro_target_to_source_center_margin'],'same_class_source_target_centroid_cosine':s['source_probe_train_centroid_target_geometry']['macro_matched_center_cosine'],'probe_converged':s['probe_converged'],'dimension':s['domain_stats']['descriptor_dimension']})
  rows+=rr;stages[stage]={'per_seed':rr,'aggregate':{key:stats([r[key] for r in rr]) for key in rr[0] if key not in ['seed','stage','probe_converged','dimension']},'all_probes_converged':all(r['probe_converged'] for r in rr)}
 neighbors={}
 for group in ['all_nonreflected','boundary3x3','interior3x3_fully_labeled']:
  rr=[results[seed]['neighbor_semantics'][group] for seed in SEEDS];neighbors[group]={'per_seed':{str(seed):results[seed]['neighbor_semantics'][group] for seed in SEEDS},'aggregate':{key:stats([r[key] for r in rr]) for key in ['mean_cosine_gap','balanced_sample_cosine_AUROC','top10_known_same_class_ratio','all_known_neighbors_same_class_ratio','top10_known_coverage','top10_same_minus_different_cosine','classbalanced_paired_patch_gap_mean']},'same_cosine_mean':stats([r['same_cosine']['mean'] for r in rr]),'different_cosine_mean':stats([r['different_cosine']['mean'] for r in rr]),'same_cosine_variance':stats([r['same_cosine']['variance'] for r in rr]),'different_cosine_variance':stats([r['different_cosine']['variance'] for r in rr])}
 transitions=[]
 for a,b in [('F3D','V'),('V','T'),('T','Z'),('F3D_mean','V_mean')]:
  for seed in SEEDS:
   aa=next(r for r in rows if r['seed']==seed and r['stage']==a);bb=next(r for r in rows if r['seed']==seed and r['stage']==b);transitions.append({'seed':seed,'transition':a+'->'+b,'delta_target_probe_OA':bb['target_probe_OA']-aa['target_probe_OA'],'delta_target_probe_AA':bb['target_probe_AA']-aa['target_probe_AA'],'delta_target_class_margin':bb['target_class_margin']-aa['target_class_margin'],'delta_target_to_source_centroid_margin':bb['target_to_source_class_margin']-aa['target_to_source_class_margin']})
 with (folder/'stage_probe_and_margin.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (folder/'stage_transitions.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(transitions[0]));w.writeheader();w.writerows(transitions)
 layerrows=[]
 for seed,r in results.items():
  for layer,v in r['CNN_layer_stats'].items():
   layerrows.append({'seed':seed,'layer':layer,'source_fullmap_norm':v['source_fullmap_stats']['full_map_norm_mean'],'target_fullmap_norm':v['target_fullmap_stats']['full_map_norm_mean'],'source_channel_var':v['source_fullmap_stats']['channel_variance_mean'],'target_channel_var':v['target_fullmap_stats']['channel_variance_mean'],**v['center_descriptor_domain_stats'],'source_margin':v['center_descriptor_geometry']['source']['macro_margin_LOO'],'target_margin':v['center_descriptor_geometry']['target']['macro_margin_LOO']})
 with (folder/'CNN_layer_stats.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(layerrows[0]));w.writeheader();w.writerows(layerrows)
 neighborrows=[]
 for group,value in neighbors.items():
  for seed,r in value['per_seed'].items():neighborrows.append({'seed':seed,'group':group,'patches':r['patches'],'same_cosine_mean':r['same_cosine']['mean'],'same_cosine_variance':r['same_cosine']['variance'],'different_cosine_mean':r['different_cosine']['mean'],'different_cosine_variance':r['different_cosine']['variance'],**{key:r[key] for key in ['mean_cosine_gap','balanced_sample_cosine_AUROC','top10_known_same_class_ratio','all_known_neighbors_same_class_ratio','top10_known_coverage','top10_same_minus_different_cosine']}})
 with (folder/'boundary_and_neighbor.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(neighborrows[0]));w.writeheader();w.writerows(neighborrows)
 dump(folder/'summary.json',{'seeds':SEEDS,'epoch':200,'primary_representation':'F3D/V center,Tflatten,Z;spatialmean auxiliary','stages':stages,'neighbor_semantics':neighbors,'transitions':transitions,'CNN_layers':layerrows,'all_checkpoints_and_running_stats_unchanged':all(r['state_and_checkpoint_unchanged'] for r in results.values()),'source_only_probe_fit_target_offline_only':True,'recommendation':'pending evidence interpretation; no automatic module design'})
 text=['Houston formal C epoch200, seeds2100/2101/2102, read-only frozen stem/value diagnosis.','Fixed source2520,target52901; fixed source probe train2393/heldout127 using seed2100 partition. Heldout is probe-only; CNN splits differ across representationseeds.','PrimaryF3D center384,Vcenter64,Tflatten256,Z64;auxF3Dmean/Vmean. Probe source-only StandardScaler/LogisticRegression C1 fixed, no target tuning.','Stage table '+json.dumps(stages),'Transitions '+json.dumps(transitions),'Neighbor analysis '+json.dumps(neighbors),'F3D includes spectral-spatial3x3; stages differ in dimension/receptivefield. Observational probes do not prove a causal encoder/BN defect.','Checkpoint weights/BN statistics unchanged. No new backbone,loss,BN variant/Transformer/Pavia modification.']
 (folder/'RESULTS.txt').write_text('\n'.join(text)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4));names=['F3D','V','T','Z'];xs=np.arange(4)
 for seed in SEEDS:
  rr=[next(r for r in rows if r['seed']==seed and r['stage']==n) for n in names];axes[0].plot(xs,[r['target_probe_OA'] for r in rr],'-o',label=str(seed));axes[1].plot(xs,[r['target_class_margin'] for r in rr],'-o',label=str(seed))
 axes[0].set_ylabel('Target source-trained probe OA (%)');axes[1].set_ylabel('Target cosine class margin')
 for ax in axes[:2]:ax.set_xticks(xs,names);ax.legend();ax.grid(alpha=.2)
 x=np.arange(3);groups=['all_nonreflected','boundary3x3','interior3x3_fully_labeled'];same=[neighbors[g]['same_cosine_mean']['mean'] for g in groups];diff=[neighbors[g]['different_cosine_mean']['mean'] for g in groups];axes[2].bar(x-.17,same,.34,label='same class');axes[2].bar(x+.17,diff,.34,label='different class');axes[2].set_xticks(x,['all','boundary','interior']);axes[2].set_ylabel('Neighbor value cosine');axes[2].legend();fig.tight_layout();fig.savefig(folder/'stem_value_diagnosis.png',dpi=180);fig.savefig(folder/'stem_value_diagnosis.svg');plt.close(fig);print('Stem/value report complete',flush=True)
if __name__=='__main__':main()
