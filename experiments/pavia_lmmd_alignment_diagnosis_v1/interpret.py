"""Compact evidence tables/figure from completed read-only artifacts; no model forwards."""
import csv
import json
from pathlib import Path
import numpy as np
import diagnostics as d

def mean(v):return float(np.mean(v))

def main():
    assert json.loads((d.OUT/'launcher_status.json').read_text())['state']=='completed_no_followups'
    s=json.loads((d.OUT/'summary.json').read_text());final={};mode_rows=[];oracle_rows=[];reliability_rows=[]
    for seed in d.SEEDS:
        r=json.loads((d.OUT/('seed_%d'%seed)/'LMMD100.json').read_text());final[seed]=r
        for k in (3,4,6,7):
            p=r['full_population_reliability_epoch100']['classes'][k-1]
            w=s['native_membership_summary']['100'][str(k)]['per_seed']
            reliability_rows.append({'seed':seed,'class':k,'population_recall':p['recall_percent'],
                'population_pseudo_precision':p['pseudo_precision_percent'],
                'native_training_weight_purity':next(v['active_weight_purity'] for v in w if v['seed']==seed),
                'wrong_high_confidence_fraction':p['wrong_high_confidence_ge_09_fraction']})
        q=r['train_mode_diagnosis'];cm=q['reliability']['confusion_matrix'];pred3=sum(v[2] for v in cm)
        r['pred3_GT4_GT7_fraction']=(cm[3][2]+cm[6][2])/pred3
        for k in (4,7):
            rows=[v for v in q['feature_gradient_samples'] if v['GT_class']==k and not v['pseudo_correct']]
            cos=np.array([v['actual_vs_oracle_feature_gradient_cosine'] for v in rows],float)
            dots=np.array([v['actual_gradient_towards_true_source_centroid_dot'] for v in rows],float)
            oracle_rows.append({'seed':seed,'class':k,'wrong_pseudo_n':len(rows),'oracle_cosine_mean':cos.mean(),
                'opposite_oracle_fraction':mean(cos<0),'moves_away_true_source_centroid_fraction':mean(dots<0)})
        folder=d.OUT/('seed_%d'%seed)
        with np.load(folder/'features_CE100.npz') as z:ce_f=z['target'];ce_p=d.probabilities(z['target_logits'])
        with np.load(folder/'features_LMMD100.npz') as z:lm_f=z['target'];lm_p=d.probabilities(z['target_logits'])
        with np.load(d.OUT/'fixed_samples.npz') as z:y=z['target_labels']
        labels=np.load(folder/'fixed_CE_target_modes.npy')
        for k in (4,7):
            original=[v for v in json.loads((folder/'CE100.json').read_text())['fixed_CE_mode_statistics'] if v['class']==k]
            easy=max(original,key=lambda v:v['recall_percent'])['mode']
            for j in (0,1):
                mask=(y==k-1)&(labels==j)
                mode_rows.append({'seed':seed,'class':k,'CE_cluster':j,'CE_reference_group':'easier' if j==easy else 'harder',
                    'n':int(mask.sum()),'CE_recall':100*mean(ce_p[mask].argmax(1)==k-1),
                    'LMMD_recall':100*mean(lm_p[mask].argmax(1)==k-1),
                    'recall_delta':100*(mean(lm_p[mask].argmax(1)==k-1)-mean(ce_p[mask].argmax(1)==k-1))})
    for name,rows in [('compact_reliability.csv',reliability_rows),('wrong_subgroup_oracle_by_seed.csv',oracle_rows),('fixed_CE_mode_recall_by_seed.csv',mode_rows)]:
        with (d.OUT/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(d.clean(rows))
    compact={'population_reliability':reliability_rows,'wrong_subgroup_oracle':oracle_rows,
        'fixed_CE_mode_recall':mode_rows,'native_class3_weight_purity':s['native_membership_summary']['100']['3'],
        'training_pred3_GT4_GT7_fraction':[final[seed]['pred3_GT4_GT7_fraction'] for seed in d.SEEDS],
        'analysis_script_sha256':d.sha(__file__),'no_new_model_or_loss':True}
    d.write(d.OUT/'compact_evidence.json',compact)
    lines=['Pavia C+CE+LMMD read-only class-aware alignment: final interpretation',
        'Baseline remains the fixedepoch100 formal87.7097OA±2.6323,90.3401AA±1.7458. Diagnostics do not select a checkpoint.',
        '3seeds2100/2101/2102;phases1/10/50/100;all original/model hashes unchanged;no optimizer or retraining.',
        '', '1. Actual class weights have specific contamination,not uniformly unreliable predictions.']
    for k in (3,4,6,7):
        rows=[v for v in reliability_rows if v['class']==k]
        lines.append('C%d: formal-population recall%.4f%%,precision%.4f%%;native training normalized weight purity%.4f%%;seed purities=%s.'%(k,
            mean([v['population_recall'] for v in rows]),mean([v['population_pseudo_precision'] for v in rows]),
            100*mean([v['native_training_weight_purity'] for v in rows]),[v['native_training_weight_purity'] for v in rows]))
    lines+=['C3 is heavily contaminated in all3seeds;many wrongly predicted trueC4/C7 enter its target cohort.',
        'Training predictedC3 cohort fraction contributed by trueC4/C7 perseed: '+str(compact['training_pred3_GT4_GT7_fraction']),
        'High precision for predictedC4/C7 does not imply adequate trueC4/C7 coverage;wrong-class high confidence makes simple confidence filtering an unproven fix.',
        '', '2. Global true-class correspondence improved;residual failure is localized.']
    for k in ('4','7'):
        a=s['aggregate']['CE100']['classes'][k];b=s['aggregate']['LMMD100']['classes'][k]
        lines.append('C%s same-class centroid squaredL2 CE%.6f→LMMD%.6f;LMMD nearest source class%s;negative correct-source sample margins%.4f%%.'%(k,
            a['source_target_same_class_centroid_squared_L2']['mean'],b['source_target_same_class_centroid_squared_L2']['mean'],
            b['nearest_source_for_target_centroid'],100*b['target_sample_negative_correct_source_margin_fraction']['mean']))
    lines+=['Class means are correctly matched in all3seeds;do not claim wholesale wrong-class center alignment.',
        'Most wrongly predictedC4/C7 are also nearest the wrong source center,especiallyC3,in the feature space.',
        '', '3. Gradient conflict exists locally,not as a persistent damaging global conflict.']
    for e in ('1','10','50','100'):
        q=s['gradient_summary'][e]['all_parameters'];lines.append('epoch%s:meanCE–LMMD cosine%.6f;negative%.4f%%;weightedLMMD/CE norm%.6f;combined first-order source-CE uphill%.4f%%.'%(e,
            q['cosine_mean']['mean'],100*q['negative_cosine_fraction']['mean'],q['weighted_norm_ratio_mean']['mean'],100*q['CE_first_order_uphill_fraction']['mean']))
    lines+=['Late Transformer directions are often opposed,while observed global combined directions remain source-CE descent. Scalar norm ratios do not prove a need for per-class lambdas.',
        'These are frozen end-of-epoch gradient queries,with exact historical inputs/dropout,not unsaved actual per-update gradients;momentum and weight decay are excluded from the dot diagnostic.',
        '', '4. GT-weight oracle isolates direction errors on wrong subgroups,without training.']
    for k in ('4','7'):
        x=s['target_feature_gradient_oracle_summary']['100'][k]
        a=x['pseudo_correct_True'];b=x['pseudo_correct_False']
        lines.append('C%s actual-vs-oracle target feature gradient cosine:correct%.6f(n%d),wrong%.6f(n%d);wrong opposite%.4f%%;away-from-true-source-center%.4f%%.'%(k,
            a['oracle_cosine']['mean'],a['n'],b['oracle_cosine']['mean'],b['n'],100*b['opposite_oracle_fraction'],100*b['moves_away_from_true_source_centroid_fraction']))
    lines+=['C7 wrong-subgroup mean cosine is negative in all3seeds;C4 is mixed across seeds. Small sampled wrong subgroups are diagnostic,not a target accuracy oracle result.',
        '', '5. Within-class tradeoff is consistent forC7;semantic overcompression remains unproven.']
    for seed in d.SEEDS:
        chosen=[v for v in mode_rows if v['seed']==seed and v['class']==7]
        lines.append('seed%d C7 fixedCE feature groups: '%seed+str(chosen))
    lines+=['Both fixedC4 groups improve in all3seeds. C7 has one formerly easier group losing recall in eachseed,while the other improves.',
        'C7 between-group squareddistance LMMD/CE ratios: '+str([v['between_mode_distance_ratio'] for v in s['fixed_mode_CE_vs_LMMD'] if v['class']==7 and v['mode']==0]),
        'Two CE feature clusters are descriptive groups;no evidence yet that preserving their exact geometry is necessary or that compression itself causes harm.',
        '', 'Recommended bottleneck priority:',
        'Reliable class correspondence for misclassified subgroups,especiallyC4/C7 assignedC3;protect already useful within-class groups while correcting missed ones.',
        'Class-specific/global strength tuning and a claim of semantic multimodal overcompression are not established by this diagnosis.',
        'Relation-aware information has not yet been shown to correct these errors;no new relation loss,module,teacher or target-dependent training was designed/launched.',
        'GT was used only offline for audits/counterfactual gradients. All15 checkpoint analyses preserved state;all original checkpoints/data hashes remain unchanged.']
    (d.OUT/'FINAL_INTERPRETATION.txt').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    fig,axs=plt.subplots(2,2,figsize=(12,8));epochs=np.array(d.PHASES)
    for k in (3,4,6,7):
        rr=[s['native_membership_summary'][str(e)][str(k)]['active_weight_purity'] for e in epochs]
        m=np.array([v['mean'] for v in rr]);sd=np.array([v['sample_std'] for v in rr])
        axs[0,0].plot(epochs,100*m,'o-',label='C%d'%k);axs[0,0].fill_between(epochs,100*(m-sd),100*(m+sd),alpha=.1)
    axs[0,0].set(xscale='log',xticks=epochs,xticklabels=epochs,ylim=(0,105),ylabel='Native class-weight purity (%)',xlabel='Checkpoint epoch',title='Training weights: mean ± SD, 3 seeds');axs[0,0].legend()
    xx=np.arange(2)
    for kind,key,offset in [('CE','CE100',-.18),('CE+LMMD','LMMD100',.18)]:
        vals=[s['aggregate'][key]['classes'][k]['source_target_same_class_centroid_squared_L2'] for k in ('4','7')]
        axs[0,1].bar(xx+offset,[v['mean'] for v in vals],width=.36,yerr=[v['sample_std'] for v in vals],capsize=3,label=kind)
    axs[0,1].set(xticks=xx,xticklabels=['C4','C7'],ylabel='Squared L2, actual CLS64',title='Same-class source–target centroid distance');axs[0,1].legend()
    for correct,offset,label in [(True,-.18,'Correct pseudo'),(False,.18,'Wrong pseudo')]:
        values=[]
        for k in (4,7):
            per=[]
            for seed in d.SEEDS:
                rows=[r for r in final[seed]['train_mode_diagnosis']['feature_gradient_samples'] if r['GT_class']==k and r['pseudo_correct']==correct]
                per.append(mean([r['actual_vs_oracle_feature_gradient_cosine'] for r in rows]))
            values.append(per)
        axs[1,0].bar(xx+offset,[mean(v) for v in values],width=.36,yerr=[np.std(v,ddof=1) for v in values],capsize=3,label=label)
    axs[1,0].axhline(0,color='k',linewidth=.8);axs[1,0].set(xticks=xx,xticklabels=['C4','C7'],ylabel='Feature-gradient cosine',title='Actual vs GT-weight oracle, frozen epoch100');axs[1,0].legend()
    for group,off in [('easier',-.18),('harder',.18)]:
        vals=[next(v['recall_delta'] for v in mode_rows if v['seed']==seed and v['class']==7 and v['CE_reference_group']==group) for seed in d.SEEDS]
        axs[1,1].bar(np.arange(3)+off,vals,width=.36,label='CE '+group+' group')
    axs[1,1].axhline(0,color='k',linewidth=.8);axs[1,1].set(xticks=np.arange(3),xticklabels=d.SEEDS,ylabel='Recall delta: LMMD − CE (pp)',title='C7: fixed CE feature groups (descriptive)');axs[1,1].legend()
    fig.suptitle('Pavia: read-only class-aware LMMD diagnosis',fontsize=14);fig.tight_layout()
    fig.savefig(d.OUT/'alignment_diagnosis.png',dpi=180);fig.savefig(d.OUT/'alignment_diagnosis.svg');plt.close(fig)
    print('Compact evidence,interpretation and standalone figure saved.',flush=True)

if __name__=='__main__':main()
