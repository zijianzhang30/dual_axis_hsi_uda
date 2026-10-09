import json,csv
from pathlib import Path
import numpy as np
from common import OUT,SEEDS,dump,sha

a=json.loads((OUT/'summary.json').read_text());distance_rows=[];class_summary=[];c6_table=[]
for seed in SEEDS:
 for arm in ['A','C']:
  mask=np.load(OUT/f'seed_{seed}/fixed_C6_masks_and_coordinates.npz');r=a['runs'][str(seed)][arm]
  for stage in ['T','Z']:
   scores=np.load(OUT/f'seed_{seed}/{arm}_{stage}_source_centroid_scores.npz')['target_scores']
   for kind in ['all_target_C6',f'{arm}_final_C6_to_C5','C_final_C6_to_C5']:
    selected=mask[kind];s=scores[selected]
    distance_rows.append({'seed':seed,'arm':arm,'stage':stage,'subset':kind,'n':int(selected.sum()),'mean_cosine_to_source_C5':float(s[:,4].mean()),'mean_cosine_to_source_C6':float(s[:,5].mean()),'mean_cosine_distance_to_source_C5':float(1-s[:,4].mean()),'mean_cosine_distance_to_source_C6':float(1-s[:,5].mean()),'mean_C6_minus_C5_cosine_margin':float((s[:,5]-s[:,4]).mean())})
 for stage in ['T','Z']:
  pass
 c=a['runs'][str(seed)]['C'];d=c['C6_subsets']['own_final_C6_to_C5'];allc=c['C6_subsets']['all_target_C6']
 c6_table.append({'seed':seed,'final_C6_to_C5_n':d['n'],'error_T_nearest_source_C5_percent':100*d['T_nearest_source_C5_fraction'],'error_Z_nearest_source_C5_percent':100*d['Z_nearest_source_C5_fraction'],'error_strict_T_sourceC6_to_Z_sourceC5_percent':100*d['nearest_source_C6_to_C5_n']/d['n'],'all_C6_T_nearest_source_C6_percent':100*allc['T_nearest_source_C6_fraction'],'all_C6_Z_nearest_source_C6_percent':100*allc['Z_nearest_source_C6_fraction'],'all_C6_source_C5_to_C6_rescue_count':allc['nearest_source_C5_to_C6_n'],'all_C6_source_C6_to_C5_harm_count':allc['nearest_source_C6_to_C5_n']})
for stage in ['T','Z']:
 for c in range(7):
  deltas=[];aa=[];cc=[];margins=[]
  for seed in SEEDS:
   x=a['runs'][str(seed)]['A']['stages'][stage]['per_class'][c];y=a['runs'][str(seed)]['C']['stages'][stage]['per_class'][c];v=100*(y['per_sample_correct_source_centroid_fraction']-x['per_sample_correct_source_centroid_fraction']);deltas.append(v);aa.append(100*x['per_sample_correct_source_centroid_fraction']);cc.append(100*y['per_sample_correct_source_centroid_fraction']);margins.append(y['mean_correct_vs_bestwrong_margin']-x['mean_correct_vs_bestwrong_margin'])
  class_summary.append({'stage':stage,'target_class':c+1,'A_correct_source_nearest_mean_percent':float(np.mean(aa)),'C_correct_source_nearest_mean_percent':float(np.mean(cc)),'C_minus_A_mean_pp':float(np.mean(deltas)),'C_minus_A_sample_std_pp':float(np.std(deltas,ddof=1)),'seed2100_delta_pp':deltas[0],'seed2101_delta_pp':deltas[1],'seed2102_delta_pp':deltas[2],'positive_seeds':sum(v>0 for v in deltas),'negative_seeds':sum(v<0 for v in deltas),'mean_C_minus_A_margin':float(np.mean(margins))})
for name,rows in [('C6_source_C5_C6_distances.csv',distance_rows),('C6_key_results_by_seed.csv',c6_table),('class_correspondence_C_minus_A_mean_std.csv',class_summary)]:
 with (OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['Houston A/C class-conditional T/Z geometry: fixed3seeds epoch200','Main answer: C6->C5 confusion is usually already present at T, not mainly introduced by T->Z.','Centroids use only each model original2393 source training samples; same target52901 pixels; no new training/module/head or target tuning.','', 'C final C6->C5 subgroup, per seed:']
for r in c6_table:lines.append(json.dumps(r))
p=a['pooled_C6_subset_transitions']['C'];lines+=['Pooled final error subgroup: T nearestsourceC5 %.4fpercent; Z nearestsourceC5 %.4fpercent; strictTnearestC6->ZnearestC5 %.4fpercent.'%(100*p['own_final_C6_to_C5']['T_already_nearest_source_C5_fraction'],100*p['own_final_C6_to_C5']['Z_nearest_source_C5_fraction'],100*p['own_final_C6_to_C5']['strict_source_C6_T_to_source_C5_Z_fraction']),'Pooled here repeats same pixels across3seed models; not independent sample size.','All C6 correctly closest sourceC6: C T76.37615percent -> Z85.80668percent; A T72.92338percent -> Z83.79721percent. Both improve in everyseed.','For all C6 in C: strictnearest5->6 rescues6808 vs6->5 harms203, across3seed observations. Conditional finalerrors alone must not be used to infer Transformer harms allC6.','', 'Whole-class means: C6 mean nearestsourceC6 at BOTH stages in ALL6(A/Cx3seed) models; hard errors are a subgroup, not wholesale C6 class-center shift.','Only whole-class mismatch: targetC7 nearestsourceC6 for A seed2101 atT/Z, and C seed2101 atT; C Z corrects tosourceC7. All other wholeclass means match ownsourceclass.','', 'C-A per-sample correctsourcecentroid correspondence gains:']
for r in class_summary:lines.append(json.dumps(r))
lines+=['C improves C2/C3 atT/Z all3 seeds, C6atT all3seeds; C6atZ improves2/3. C1atT worsensall3; C5atZ worsensall3. Not every class correspondence improves.','C6->C5 native final error counts A=[3700,2764,2797], C=[2179,2995,3219]; C reduces this specific error only seed2100, although general correspondence gains can coexist.','Sensitivity: Tmean vsflatten, and unitmean-sourcecentroid vsrawmean, preserve the conclusion that most finalC6->C5 pixels already matchsourceC5atT.','Interpretation: locate main remaining confusion no later than token formation, not primarily in Transformer class-geometry preservation. Do not automatically create a Transformer module.','T already wrong does NOT establish tokenizer itself is cause: CNN values may already be mixed, and prototypes approximate multimodal geometry. A/C have jointly retrained CNN/Transformer weights.','Token class discrimination at/beforeT is a candidate analysis location, not validated second innovation. No architecture design or new experiment triggered.','All distances cosine distance1-cos; descriptorT256 vsZ64 scales differ; nearestclass/signs are primary evidence. Crossstagecosine margins are descriptive, not causal.','All checkpoint/cache hashes unchanged; Pavia untouched.']
(OUT/'FINAL_INTERPRETATION.txt').write_text('\n'.join(lines)+'\n')
dump(OUT/'CONCLUSIONS.json',{'C6_to_C5_mainly_present_at_T_supported_all3':True,'Transformer_main_harm_hypothesis_supported':False,'whole_C6_centroid_matches_source_C6_all_models_stages':True,'tokenizer_alone_causally_responsible':False,'new_module_or_training':False,'Pavia_modified':False,'next_location_if_further_diagnosis':'at_or_before_T; no automatic design'})
# Standalone scientific figure.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.6));colors=['#d47b25','#2d72b8','#a7a7a7'];x=np.arange(6);bottom=np.zeros(6)
vals=[]
for seed in SEEDS:
 r=a['runs'][str(seed)]['C']['C6_subsets']['own_final_C6_to_C5']
 for stage in ['T','Z']:
  c5=r[stage+'_nearest_source_C5_fraction'];c6=r[stage+'_nearest_source_C6_fraction'];vals.append([100*c5,100*c6,100*(1-c5-c6)])
vals=np.asarray(vals)
for i,label in enumerate(['Nearest source C5','Nearest source C6','Other source class']):axes[0].bar(x,vals[:,i],bottom=bottom,color=colors[i],label=label);bottom+=vals[:,i]
axes[0].set_xticks(x);axes[0].set_xticklabels(['2100 T','2100 Z','2101 T','2101 Z','2102 T','2102 Z']);axes[0].set_ylim(0,105);axes[0].set_ylabel('Fraction of final C6-to-C5 errors (%)');axes[0].set_title('C: confusion is already present at T');axes[0].legend(loc='lower left',fontsize=8)
labels=[];means=[];stds=[]
for arm in ['A','C']:
 for stage in ['T','Z']:
  v=[100*a['runs'][str(s)][arm]['C6_subsets']['all_target_C6'][stage+'_nearest_source_C6_fraction'] for s in SEEDS];labels.append(arm+' '+stage);means.append(np.mean(v));stds.append(np.std(v,ddof=1))
axes[1].bar(np.arange(4),means,yerr=stds,capsize=5,color=['#aaa','#777','#78a9d5','#2d72b8']);axes[1].set_xticks(np.arange(4));axes[1].set_xticklabels(labels);axes[1].set_ylim(0,100);axes[1].set_ylabel('Closest source class is C6 (%)');axes[1].set_title('All target C6: T-to-Z improves correspondence')
fig.suptitle('Houston fixed epoch200, three seeds; cosine source-centroid geometry');fig.tight_layout();fig.savefig(OUT/'C6_T_Z_geometry.png',dpi=200);fig.savefig(OUT/'C6_T_Z_geometry.svg');plt.close(fig)
p=json.loads((OUT/'provenance.json').read_text());p['report_code_sha256']=sha(OUT/'report.py');dump(OUT/'provenance.json',p)
dump(OUT/'status.json',{'state':'completed_readonly','fixed_seeds':SEEDS,'fixed_epoch':200,'new_training':False,'no_auto_modules':True,'Pavia_modified':False})
print('Class geometry report complete')
