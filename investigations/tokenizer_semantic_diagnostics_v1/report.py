import pathlib,json,numpy as np,torch,csv,itertools,hashlib
OUT=pathlib.Path(__file__).parent;BASE=OUT.parent/'tokenizer_mechanism_v2';SEEDS=[2100,2101,2102]
def stats(v):
 values=[None if x is None or not np.isfinite(x) else float(x) for x in v];x=np.array([z for z in values if z is not None],float)
 return {'values':values,'valid_seed_count':len(x),'mean':float(x.mean()) if len(x) else None,'sample_std':float(x.std(ddof=1)) if len(x)>1 else None}
def fmt(v):return 'NA' if v is None else f'{v:.6f}'
def learned_margin(seed,arm,domain):
 f=OUT/f'seed_{seed}';z=np.load(f/f'{arm}_{domain}_Z.npy').astype(float);labels=np.load(f/f'{domain}_labels_offline.npy');state=torch.load(BASE/'houston'/f'{arm}_{seed}/checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model'];w=state['nn1.weight'].numpy().astype(float);b=state['nn1.bias'].numpy().astype(float);logits=z@w.T+b
 distance=np.linalg.norm(w[:,None,:]-w[None,:,:],axis=-1);den=distance[labels];diff=logits[np.arange(len(labels)),labels,None]-logits;np.maximum(den,1e-30,out=den);margin=diff/den;margin[np.arange(len(labels)),labels]=np.inf;margin=margin.min(1)
 return {'macro_geometric_classifier_margin':float(np.mean([margin[labels==c].mean() for c in range(7)])),'micro_geometric_classifier_margin':float(margin.mean()),'per_class':{str(c+1):{'n':int((labels==c).sum()),'geometric_margin_mean':float(margin[labels==c].mean()),'positive_fraction':float((margin[labels==c]>0).mean())} for c in range(7)}}
def main():
 rows={s:json.loads((OUT/f'seed_{s}/summary.json').read_text()) for s in SEEDS};levels=[q['strength'] for q in rows[2100]['A']['affine_curve']];curve={};paired={};retention={}
 for arm in 'AC':
  curve[arm]={str(level):{m:stats([rows[s][arm]['affine_curve'][i]['metrics'][m] for s in SEEDS]) for m in ['oa_percent','aa_percent','kappa']} for i,level in enumerate(levels)}
  retention[arm]={}
  for m in ['oa_percent','aa_percent']:
   values=[];loss=[]
   for s in SEEDS:
    score=np.array([q['metrics'][m] for q in rows[s][arm]['affine_curve']]);values.append(float(np.trapz(score/score[0],levels)/levels[-1]));loss.append(float(np.trapz(score[0]-score,levels)/levels[-1]))
   retention[arm][m]={'normalized_retention_AUC':stats(values),'mean_integrated_drop_pp':stats(loss)}
 for i,level in enumerate(levels):
  paired[str(level)]={}
  for metric in ['oa_percent','aa_percent']:
   paired[str(level)][metric]={'C_minus_A_score':stats([rows[s]['C']['affine_curve'][i]['metrics'][metric]-rows[s]['A']['affine_curve'][i]['metrics'][metric] for s in SEEDS]),'C_minus_A_own_clean_drop':stats([(rows[s]['C']['affine_curve'][0]['metrics'][metric]-rows[s]['C']['affine_curve'][i]['metrics'][metric])-(rows[s]['A']['affine_curve'][0]['metrics'][metric]-rows[s]['A']['affine_curve'][i]['metrics'][metric]) for s in SEEDS])}
 select={};select_delta={};sep={};sep_delta={};classifier={}
 for arm in 'AC':
  select[arm]={};sep[arm]={};classifier[arm]={}
  for dom in ['source','target']:
   select[arm][dom]={group:{k:stats([rows[s][arm]['spatial_selectivity'][dom][group]['macro'][k] for s in SEEDS]) for k in ['conditional_same_class_mass_excluding_center','excess_over_uniform_known_neighbors','center_weight','raw_same_class_mass_excluding_center','known_neighbor_weight_mass_excluding_center','known_neighbor_fraction']} for group in ['all_no_reflection','boundary3x3','interior3x3_fully_labeled','mixed13x13']}
   cs=[learned_margin(s,arm,dom) for s in SEEDS];classifier[arm][dom]={'macro_geometric_margin':stats([v['macro_geometric_classifier_margin'] for v in cs]),'micro_geometric_margin':stats([v['micro_geometric_classifier_margin'] for v in cs]),'per_seed':{str(s):v for s,v in zip(SEEDS,cs)}}
  for stage in ['T','Z']:
   sep[arm][stage]={}
   for dom in ['source','target','source_to_target']:
    metrics=['macro_within_class_cosine_distance','macro_nearest_center_distance','macro_margin_LOO','macro_positive_margin_fraction'] if dom!='source_to_target' else ['macro_matched_center_cosine','macro_matched_center_margin','macro_target_to_source_center_margin','macro_source_prototype_target_recall']
    sep[arm][stage][dom]={k:stats([rows[s][arm]['class_separability'][stage][dom][k] for s in SEEDS]) for k in metrics}
 for dom in ['source','target']:
  select_delta[dom]={group:{metric:stats([c-a if c is not None and a is not None else None for c,a in zip(select['C'][dom][group][metric]['values'],select['A'][dom][group][metric]['values'])]) for metric in select['A'][dom][group]} for group in select['A'][dom]}
 for stage in ['T','Z']:
  sep_delta[stage]={dom:{metric:stats(np.asarray(sep['C'][stage][dom][metric]['values'])-sep['A'][stage][dom][metric]['values']) for metric in sep['A'][stage][dom]} for dom in sep['A'][stage]}
 result={'affine_curves':curve,'affine_paired_C_minus_A':paired,'affine_retention':retention,'selectivity_macro':select,'selectivity_paired_C_minus_A':select_delta,'class_separability':sep,'separability_paired_C_minus_A':sep_delta,'learned_classifier_margin':classifier,'all_per_seed_and_per_class_records':{str(s):rows[s] for s in SEEDS},'no_training':True,'GT_offline_only':True,'no_automatic_new_modules':True}
 (OUT/'REPORT_DATA.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['Houston read-only A/C three mechanism diagnostics, fixed epoch200,3seeds','GT usedonlyoffline. Source/target indices fullnativefixed acrossmodels. No extra training orPavia modifications. Source has0 observed3x3 mixedclass boundary patches afterreflectionexclusion, so sourceboundary NA; source mixed13x13 group142patches. Target observed3x3 boundary1836patches.','Affine stressafter normband;gain=exp(s*u_band),offset=.1*s*v_band;oneRNG94000 vector allpatches/allseeds. No clipping/renormalization.','Allzero-strength predicted labels andlogits match official results.','', 'Affine curve (mean+/-sampleSD,OA/AA percent):']
 for level in levels:
  parts=[]
  for arm in 'AC':
   z=curve[arm][str(level)];parts.append(arm+' '+', '.join(f"{k} {z[k]['mean']:.3f}+/-{z[k]['sample_std']:.3f}" for k in ['oa_percent','aa_percent']))
  lines.append(str(level)+' '+ '; '.join(parts))
 lines+=['RetentionAUC (normalizedbyownclean): '+json.dumps(retention),'','Spatialselectivity: knownneighbors only,centerexcluded,reflection excluded;GT undefined notwrongclass. Macroperclass primary. Boundary=observed3x3 otherclass;interior=fullyknown3x3 sameclass.']
 for dom in ['source','target']:
  for group in ['all_no_reflection','boundary3x3','interior3x3_fully_labeled']:
   for metric in ['conditional_same_class_mass_excluding_center','excess_over_uniform_known_neighbors','raw_same_class_mass_excluding_center','known_neighbor_weight_mass_excluding_center']:
    a=select['A'][dom][group][metric];c=select['C'][dom][group][metric];lines.append(f"{dom} {group} {metric}: A{fmt(a['mean'])}+/-{fmt(a['sample_std'])};C{fmt(c['mean'])}+/-{fmt(c['sample_std'])};C-A{select_delta[dom][group][metric]['values']}")
 lines+=['','Classseparability:L2normalizedT/Z. WithinclassowncentroidLOO vsnearestothercentroid. Source-targetprototype diagnosticsGT offlineonly,nottrainedclassifier accuracy.']
 for stage in ['T','Z']:
  for dom,metrics in sep_delta[stage].items():
   for metric,values in metrics.items():lines.append(f"{stage} {dom} {metric}: A{sep['A'][stage][dom][metric]['mean']:.6f};C{sep['C'][stage][dom][metric]['mean']:.6f};C-A seeds{values['values']}")
 lines+=['Classifier geometric margin=(logity-logitj)/||wy-wj||,minoverothers; macroclasses,negative=wrongside.','Classifier margins '+json.dumps(classifier),'','Caveats: targetclass4only22samples;neighborGT oftenincomplete;highconfidenceinterior andobservedboundaries restrictedtoannotatedneighbors. Neighborpatchoverlap impliesnonindependence. A/C have differentlytrainedCNNs,classifiers,soassociation notunique causal attribution toW.','Full perclass/counts/matrices/curves inREPORT_DATA.json andseed_*/*diagnostics.json. No automaticnewmodule.']
 (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 with (OUT/'affine_curve.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['strength','arm','OAmean','OAsampleSD','AAmean','AAsampleSD'])
  for level in levels:
   for arm in 'AC':
    z=curve[arm][str(level)];w.writerow([level,arm,z['oa_percent']['mean'],z['oa_percent']['sample_std'],z['aa_percent']['mean'],z['aa_percent']['sample_std']])
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for arm,color in [('A','#64748b'),('C','#2563eb')]:
  for ax,metric in zip(axes[:2],['oa_percent','aa_percent']):
   vals=[curve[arm][str(l)][metric] for l in levels];ax.errorbar(levels,[v['mean'] for v in vals],yerr=[v['sample_std'] for v in vals],marker='o',label=arm,color=color,capsize=3);ax.set_xlabel('Affine strength');ax.set_ylabel(metric[:2].upper()+' (%)');ax.legend()
  axes[2].plot(levels,[(curve[arm][str(l)]['oa_percent']['mean']/curve[arm]['0.0']['oa_percent']['mean']) for l in levels],marker='o',label=arm,color=color)
 axes[0].set_title('Fixed-checkpoint OA');axes[1].set_title('Fixed-checkpoint AA');axes[2].set_title('OA retention relative to clean');axes[2].set_xlabel('Affine strength');axes[2].set_ylabel('Retention');axes[2].legend();fig.tight_layout();fig.savefig(OUT/'affine_robustness_curve.png',dpi=180);fig.savefig(OUT/'affine_robustness_curve.svg');plt.close(fig)
 (OUT/'status.json').write_text(json.dumps({'state':'completed','three_diagnostics':True,'seeds':SEEDS,'no_training':True,'Pavia_modified':False},indent=2)+'\n')
 print('Completed read-onlymechanism report',flush=True)
if __name__=='__main__':main()
