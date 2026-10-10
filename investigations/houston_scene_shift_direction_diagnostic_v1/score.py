"""Frozen source feature perturbation scores and fair paired comparisons."""
import json
import csv
import numpy as np
from common import OUT,SEEDS,EPS,PROBES,REPS,CONFIG,sha,dump,csvwrite,metric,aggregate

def main():
    gate=json.loads((OUT/'ALL_SOURCE_PERTURBATIONS_FROZEN.json').read_text())
    assert gate['all_seeds_probes_eps_controls_completed'] and not gate['target_GT_opened']
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    y=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy')
    masks={'all_source':np.ones(len(y),bool),'source_train':train,'source_holdout':~train}
    for c in range(7):masks[f'C{c+1}']=y==c
    rows=[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';baseline=np.load(f/'source_replayed_logits.npy')
        for probe in PROBES:
            for file in sorted((f/probe).glob('D*_eps*.npz')):
                method,tail=file.stem.rsplit('_rep',1);rep,ep=tail.split('_eps');eps=float(ep)
                a=np.load(file);logits=a['logits']
                assert np.array_equal(logits.argmax(1),a['predictions'])
                for subset,mask in masks.items():
                    rows.append({'seed':seed,'probe':probe,'method':method,'rep':int(rep),'epsilon':eps,'subset':subset,
                        **metric(logits[mask],y[mask],baseline[mask])})
            print(seed,probe,'all source metrics computed',flush=True)
    csvwrite('source_metrics_per_repetition.csv',rows)
    metrics=[k for k in rows[0] if k not in ['seed','probe','method','rep','epsilon','subset']]
    perseed=aggregate(rows,['seed','probe','method','epsilon','subset'],metrics)
    csvwrite('source_metrics_per_seed.csv',perseed)
    # Flatten repetition means; model-seed SD must not treat repetition trials as new models.
    seedmeans=[{**{key:r[key] for key in ['seed','probe','method','epsilon','subset']},
         **{key:r[key+'_mean'] for key in metrics}} for r in perseed]
    across=aggregate(seedmeans,['probe','method','epsilon','subset'],metrics)
    csvwrite('source_metrics_model_seed_aggregate.csv',across)
    csvwrite('per_class.csv',[r for r in across if r['subset'].startswith('C')])
    paired=[];flags={};classflags={}
    for probe in PROBES:
        passed=[];flipflags=[]
        for seed in SEEDS:
            for eps in EPS:
                for subset in masks:
                    d=next(r for r in perseed if r['seed']==seed and r['probe']==probe and r['method']=='D_domain' and r['epsilon']==eps and r['subset']==subset)
                    for control in ['D_random','D_orthogonal','D_domain_random','D_random_subspace_adversarial']:
                        c=next(r for r in perseed if r['seed']==seed and r['probe']==probe and r['method']==control and r['epsilon']==eps and r['subset']==subset)
                        sd=c['margin_drop_mean_sample_std'] or 0
                        paired.append({'seed':seed,'probe':probe,'epsilon':eps,'subset':subset,'control':control,
                            'domain_extra_margin_drop':d['margin_drop_mean_mean']-c['margin_drop_mean_mean'],
                            'domain_flip_increase_pp':d['flip_percent_mean']-c['flip_percent_mean'],
                            'domain_extra_OA_loss_pp':c['OA_mean']-d['OA_mean'],
                            'control_repetition_margin_drop_SD':sd,
                            'domain_drop_above_control_mean_plus2SD':d['margin_drop_mean_mean']>c['margin_drop_mean_mean']+2*sd})
                        if subset=='all_source' and control=='D_random_subspace_adversarial':
                            passed.append(d['margin_drop_mean_mean']>c['margin_drop_mean_mean']+2*sd)
                            if eps==.1:flipflags.append(d['flip_percent_mean']>c['flip_percent_mean'])
        flags[probe]=all(passed);classflags[probe]=all(flipflags)
    csvwrite('paired_direction_effects_per_seed.csv',paired)
    comparisons=aggregate(paired,['probe','epsilon','subset','control'],['domain_extra_margin_drop','domain_flip_increase_pp','domain_extra_OA_loss_pp','control_repetition_margin_drop_SD'])
    csvwrite('paired_direction_effects_model_seed_aggregate.csv',comparisons)
    semantic=list(csv.DictReader((OUT/'source_semantic_direction_audit.csv').open()))
    for r in semantic:
        for k in ['seed','rep','rank']:r[k]=int(r[k])
        for k in ['source_between_class_energy_fraction','source_within_class_energy_fraction','centered_classifier_weight_energy_fraction','isotropic_expected_weight_fraction']:r[k]=float(r[k])
    semseed=aggregate(semantic,['seed','probe','basis_type'],['source_between_class_energy_fraction','source_within_class_energy_fraction','centered_classifier_weight_energy_fraction'])
    semrows=[{**{key:r[key] for key in ['seed','probe','basis_type']},
        **{key:r[key+'_mean'] for key in ['source_between_class_energy_fraction','source_within_class_energy_fraction','centered_classifier_weight_energy_fraction']}} for r in semseed]
    semagg=aggregate(semrows,['probe','basis_type'],['source_between_class_energy_fraction','source_within_class_energy_fraction','centered_classifier_weight_energy_fraction'])
    csvwrite('semantic_direction_model_seed_aggregate.csv',semagg)
    decision='Go-for-directional-margin-diagnosis' if any(flags.values()) else 'No-Go'
    summary={'config':CONFIG,'decision':decision,'clear_directional_margin_by_probe':flags,
        'classification_flip_evidence_by_probe':classflags,'metrics_aggregate':across,'paired_effects':comparisons,
        'source_semantic_audit':semagg,'target_GT_never_accessed':True,'no_automatic_training_followup':True,
        'limits':['Domain distribution difference can reflect label proportions and semantic structure, not uniquely scene nuisance.',
          'Exact worst-domain attack is label-guided; primary isotropic random controls are not. Matched-rank adversarial control is needed.',
          'Prespecified mean+2 repetitionSD rule is descriptive, not a significance test or p-value.',
          'Unperturbed source includes training pixels; heldout127 is a same-scene spatially correlated subset.',
          'Frozen raw-Z linear-head perturbations do not establish feasible input-space scene shifts or training benefit.',
          'All eps/rank/probes/random repetitions retained; no best outcome chosen.']}
    dump(OUT/'summary.json',summary);report(summary)
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'frozen_features_directions_perturbations_head_predictions_main_checkpoints_unchanged':True,
        'target_GT_never_accessed':True,'no_training_backprop_tuning_followup':True,'decision':decision,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='SCORING_COMPLETE.json'}})

def report(s):
    def fmt(r,key):
        m=r.get(key+'_mean');sd=r.get(key+'_sample_std')
        return 'NA' if m is None else f'{m:.5f} +/- {sd:.5f}' if sd is not None else f'{m:.5f} +/- NA'
    lines=['Houston scene-shift direction V1: '+s['decision'],
        'Frozen originalC3seeds epoch200,native rawZ64/head; source2520+target52901 unlabeled directions.',
        'Global mean rank1; covariance discrepancy SVD rank4; eps.02/.05/.10;16 fixed random repetitions.',
        'No target GT access,network training,backprop,hyperparameter selection or automatic follow-up.',
        'Direction freeze precedes source GT. Source GT is used for requested exact worst margin directions.',
        'All deltas/logits/predictions frozen before scoring; rho=eps*original per-sample rawZ norm.',
        'Repetition mean within model seed first, then3modelseed mean/sampleSD. Margin in native logit units.',
        'Margin flag criterion: '+CONFIG['clear_margin_rule'],
        'Classification flag criterion: '+CONFIG['classification_rule'],
        'Margin flags: '+str(s['clear_directional_margin_by_probe']),
        'Classification flags: '+str(s['classification_flip_evidence_by_probe']),'']
    for probe in PROBES:
        lines.append('Probe '+probe)
        for eps in EPS:
            for r in s['metrics_aggregate']:
                if r['probe']==probe and r['epsilon']==eps and r['subset']=='all_source':
                    lines.append(f"eps{eps} {r['method']}: OA="+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; true margin before/after='+fmt(r,'margin_before_mean')+'/'+fmt(r,'margin_after_mean')+'; margin drop='+fmt(r,'margin_drop_mean')+'; flip='+fmt(r,'flip_percent')+'%')
            for r in s['paired_effects']:
                if r['probe']==probe and r['epsilon']==eps and r['subset']=='all_source':
                    lines.append('  vs '+r['control']+': extra margin drop='+fmt(r,'domain_extra_margin_drop')+'; flip increase='+fmt(r,'domain_flip_increase_pp')+'pp; extra OA loss='+fmt(r,'domain_extra_OA_loss_pp')+'pp')
    lines.extend(['','Source C1/C7/C6 and heldout at ALL fixed eps:'])
    for r in s['metrics_aggregate']:
        if r['subset'] in ['C1','C7','C6','source_holdout'] and r['method'] in ['D_domain','D_random_subspace_adversarial']:
            lines.append(f"{r['probe']} eps{r['epsilon']} {r['method']} {r['subset']}: OA="+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; margin drop='+fmt(r,'margin_drop_mean')+'; flip='+fmt(r,'flip_percent')+'%')
    lines.extend(['','Direction semantic audit (after source GT, does not construct or select bases):'])
    for r in s['source_semantic_audit']:lines.append(r['probe']+' '+r['basis_type']+': between-class energy='+fmt(r,'source_between_class_energy_fraction')+'; within-class='+fmt(r,'source_within_class_energy_fraction')+'; centered head-weight='+fmt(r,'centered_classifier_weight_energy_fraction'))
    lines.extend(['','Limits:',*s['limits'],'No robustness training automatically launched.'])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    methods=['D_domain','D_random','D_orthogonal','D_random_subspace_adversarial','D_domain_random']
    colors=['#e76f51','#457b9d','#2a9d8f','#9a6fb0','#e9c46a']
    for i,probe in enumerate(PROBES):
        for method,color in zip(methods,colors):
            rows=[next(r for r in s['metrics_aggregate'] if r['probe']==probe and r['epsilon']==eps and r['subset']=='all_source' and r['method']==method) for eps in EPS]
            for j,key in [(0,'margin_drop_mean'),(1,'flip_percent')]:
                axes[i,j].errorbar(EPS,[r[key+'_mean'] for r in rows],yerr=[r[key+'_sample_std'] for r in rows],marker='o',capsize=3,color=color,label=method)
        axes[i,0].set(title=probe+' true-margin drop',xlabel='Fixed epsilon / original norm',ylabel='Logit-margin drop');axes[i,0].legend(fontsize=8)
        axes[i,1].set(title=probe+' prediction flips',xlabel='Fixed epsilon / original norm',ylabel='Source samples (%)');axes[i,1].legend(fontsize=8)
    fig.suptitle('Frozen scene-discrepancy sensitivity; matched-rank adversarial control;3modelseed mean +/- SD')
    fig.savefig(OUT/'scene_shift_direction_summary.png',dpi=180);plt.close(fig)
    print('\n'.join(lines[:34]),flush=True)

if __name__=='__main__':main()
