"""Source scoring and paired before/after class-component removal."""
import csv
import json
import numpy as np
from common import HERE,BASE,OUT,SEEDS,EPS,PROBES,CONFIG,sha,dump,csvwrite,metric,aggregate

def main():
    gate=json.loads((OUT/'ALL_CLASS_ORTHOGONAL_PERTURBATIONS_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened']
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    assert sha(HERE.parent/'common.py')==prov['parent_math_sha256']
    y=np.load(OUT/'source_labels.npy');train=np.load(OUT/'source_train_mask.npy')
    masks={'all_source':np.ones(len(y),bool),'source_train':train,'source_holdout':~train}
    for c in range(7):masks[f'C{c+1}']=y==c
    rows=[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';base=np.load(f/'source_replayed_logits.npy')
        for probe in PROBES:
            for file in sorted((f/probe).glob('CO*_eps*.npz')):
                method,tail=file.stem.rsplit('_rep',1);rep,epsilon=tail.split('_eps');eps=float(epsilon)
                a=np.load(file);assert np.array_equal(a['logits'].argmax(1),a['predictions'])
                for subset,mask in masks.items():rows.append({'seed':seed,'probe':probe,'method':method,'rep':int(rep),'epsilon':eps,'subset':subset,**metric(a['logits'][mask],y[mask],base[mask])})
    csvwrite('metrics_per_repetition.csv',rows)
    mkeys=list(metric(np.eye(7),np.arange(7),np.eye(7)))
    perseed=aggregate(rows,['seed','probe','method','epsilon','subset'],mkeys)
    csvwrite('metrics_per_seed.csv',perseed)
    flattened=[{**{key:r[key] for key in ['seed','probe','method','epsilon','subset']},**{key:r[key+'_mean'] for key in mkeys}} for r in perseed]
    agg=aggregate(flattened,['probe','method','epsilon','subset'],mkeys)
    csvwrite('metrics_model_seed_aggregate.csv',agg);csvwrite('per_class.csv',[r for r in agg if r['subset'].startswith('C')])
    oldgate=json.loads((BASE/'SCORING_COMPLETE.json').read_text())
    for name in ['source_metrics_per_seed.csv','source_metrics_model_seed_aggregate.csv']:
        assert sha(BASE/name)==oldgate['files_sha256'][name]
    old=list(csv.DictReader((BASE/'source_metrics_per_seed.csv').open()))
    def oldrow(seed,probe,method,eps,subset):
        return next(r for r in old if int(r['seed'])==seed and r['probe']==probe and r['method']==method and float(r['epsilon'])==eps and r['subset']==subset)
    ratio_rows=[];flags={};status=json.loads((OUT/'projected_rank_status.json').read_text())
    for probe in PROBES:
        passes=[]
        for seed in SEEDS:
            if not status[str(seed)][probe]['valid_same_rank']:passes.append(False);continue
            for eps in EPS:
                for subset in masks:
                    d=next(r for r in perseed if r['seed']==seed and r['probe']==probe and r['method']=='CO_domain' and r['epsilon']==eps and r['subset']==subset)
                    c=next(r for r in perseed if r['seed']==seed and r['probe']==probe and r['method']=='CO_random_adversarial' and r['epsilon']==eps and r['subset']==subset)
                    od=oldrow(seed,probe,'D_domain',eps,subset);oc=oldrow(seed,probe,'D_random_subspace_adversarial',eps,subset)
                    before_d=float(od['margin_drop_mean_mean']);before_c=float(oc['margin_drop_mean_mean'])
                    after_d=d['margin_drop_mean_mean'];after_c=c['margin_drop_mean_mean'];sd=c['margin_drop_mean_sample_std'] or 0
                    before_ratio=before_d/before_c if before_c>1e-12 else None
                    after_ratio=after_d/after_c if after_c>1e-12 else None
                    passes_rule=after_d>after_c+2*sd
                    ratio_rows.append({'seed':seed,'probe':probe,'epsilon':eps,'subset':subset,
                        'before_domain_margin_drop':before_d,'before_random_margin_drop':before_c,
                        'before_domain_random_ratio':before_ratio,'after_domain_margin_drop':after_d,
                        'after_random_margin_drop':after_c,'after_random_repetition_SD':sd,
                        'after_domain_random_ratio':after_ratio,'ratio_delta':after_ratio-before_ratio if before_ratio is not None and after_ratio is not None else None,
                        'after_extra_margin_drop':after_d-after_c,
                        'after_flip_increase_pp':d['flip_percent_mean']-c['flip_percent_mean'],
                        'after_domain_OA':d['OA_mean'],'after_random_OA':c['OA_mean'],
                        'passes_original_mean_plus2SD_rule':passes_rule})
                    if subset=='all_source':passes.append(passes_rule)
        flags[probe]=bool(all(passes))
    csvwrite('before_after_ratios_per_seed.csv',ratio_rows)
    ratio_keys=['before_domain_margin_drop','before_random_margin_drop','before_domain_random_ratio','after_domain_margin_drop','after_random_margin_drop','after_random_repetition_SD','after_domain_random_ratio','ratio_delta','after_extra_margin_drop','after_flip_increase_pp','after_domain_OA','after_random_OA']
    ratioagg=aggregate(ratio_rows,['probe','epsilon','subset'],ratio_keys)
    csvwrite('before_after_ratios_model_seed_aggregate.csv',ratioagg)
    er=list(csv.DictReader((OUT/'class_energy_per_repetition.csv').open()))
    ekeys=['class_between_energy_fraction','within_class_energy_fraction','centered_classifier_weight_energy_fraction','max_class_basis_inner_product']
    for r in er:
        r['seed']=int(r['seed'])
        for key in ekeys:r[key]=float(r[key])
    es=aggregate(er,['seed','probe','method','phase'],ekeys)
    ef=[{**{key:r[key] for key in ['seed','probe','method','phase']},**{key:r[key+'_mean'] for key in ekeys}} for r in es]
    eagg=aggregate(ef,['probe','method','phase'],ekeys)
    csvwrite('class_energy_model_seed_aggregate.csv',eagg)
    assert all(r['class_between_energy_fraction']<1e-20 and r['max_class_basis_inner_product']<1e-10 for r in er if r['phase']=='class_orthogonal')
    decision='Go-for-class-orthogonal-margin-diagnosis' if any(flags.values()) else 'No-Go'
    summary={'fixed_config':CONFIG,'decision':decision,'projected_margin_evidence_by_probe':flags,
        'rank_status':status,'metrics_aggregate':agg,'before_after_ratio_aggregate':ratioagg,
        'energy_aggregate':eagg,'target_GT_never_accessed':True,'no_training_followup':True,
        'limits':['Zero source class-between energy is enforced by the empirical class-mean span; not proof that all semantic information is removed.',
          'All2520 source rows define class basis; source heldout diagnostic is not independent of basis construction.',
          'Same-rank source-label-guided attacks and paired parent random bases control attack-selection and ambient-space differences.',
          'Mean+2 repetitionSD criterion is descriptive consistency, not significance or a p-value.',
          'Raw-feature source-margin perturbations do not demonstrate target robustness-training benefit.']}
    dump(OUT/'summary.json',summary);report(summary)
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    assert sha(BASE/'FINAL_DELIVERY.json')==prov['parent_final_manifest_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'frozen_class_bases_projected_directions_attacks_checkpoints_unchanged':True,
      'parent_diagnostic_manifest_unchanged':True,'target_GT_never_accessed':True,'no_training_backprop_tuning_followup':True,
      'decision':decision,'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='SCORING_COMPLETE.json'}})

def report(s):
    def fmt(r,key):
        m=r.get(key+'_mean');sd=r.get(key+'_sample_std')
        return 'NA' if m is None else f'{m:.6g} +/- {sd:.6g}' if sd is not None else f'{m:.6g} +/- NA'
    lines=['Houston class-orthogonal control: '+s['decision'],
        'Parent scene-shift V1 unchanged; source2520,class-centroid span numeric rank<=6; no target GT.',
        'Projected domain and paired16 random subspaces share same class-orthogonal complement,rank1/4,epsilon/norm and exact worst-margin rule.',
        'ALL epsilon .02/.05/.10 retained. Repetition means within seed then3modelseed mean/sampleSD.',
        'Before/after ratio computed perseed before aggregating, never ratio of pooled means.',
        'Criterion unchanged: '+CONFIG['criterion'],
        'Projected probe flags: '+str(s['projected_margin_evidence_by_probe']),'']
    for r in s['before_after_ratio_aggregate']:
        if r['subset']=='all_source':
            lines.append(f"{r['probe']} eps{r['epsilon']}: domain/random ratio before="+fmt(r,'before_domain_random_ratio')+'; after='+fmt(r,'after_domain_random_ratio')+'; change='+fmt(r,'ratio_delta'))
            lines.append('  after domain margin drop='+fmt(r,'after_domain_margin_drop')+'; matched random='+fmt(r,'after_random_margin_drop')+'; extra='+fmt(r,'after_extra_margin_drop')+'; flip increase='+fmt(r,'after_flip_increase_pp')+'pp')
    lines.extend(['','Source all/heldout/C1/C7/C6 at ALL epsilon:'])
    for r in s['metrics_aggregate']:
        if r['subset'] in ['all_source','source_holdout','C1','C7','C6']:
            lines.append(f"{r['probe']} eps{r['epsilon']} {r['method']} {r['subset']}: OA="+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; true margin before/after='+fmt(r,'margin_before_mean')+'/'+fmt(r,'margin_after_mean')+'; drop='+fmt(r,'margin_drop_mean')+'; flip='+fmt(r,'flip_percent')+'%')
    lines.extend(['','Residual class-between and head-weight energy:'])
    for r in s['energy_aggregate']:lines.append(f"{r['probe']} {r['method']} {r['phase']}: between="+fmt(r,'class_between_energy_fraction')+'; within='+fmt(r,'within_class_energy_fraction')+'; centered head-weight='+fmt(r,'centered_classifier_weight_energy_fraction')+'; class orthogonality max='+fmt(r,'max_class_basis_inner_product'))
    lines.extend(['','Limits:',*s['limits'],'No robustness model or any further diagnostic automatically launched.'])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for probe,color in zip(PROBES,['#457b9d','#e76f51']):
        if not any(r['probe']==probe and r['subset']=='all_source' for r in s['before_after_ratio_aggregate']):continue
        rs=[next(r for r in s['before_after_ratio_aggregate'] if r['probe']==probe and r['epsilon']==eps and r['subset']=='all_source') for eps in EPS]
        for key,style,label in [('before_domain_random_ratio','o--','before'),('after_domain_random_ratio','s-','class-orthogonal')]:
            axes[0].errorbar(EPS,[r[key+'_mean'] for r in rs],yerr=[r[key+'_sample_std'] for r in rs],fmt=style,color=color,capsize=3,label=probe+' '+label)
        for key,style,label in [('after_domain_margin_drop','o-','domain'),('after_random_margin_drop','s--','paired random')]:
            axes[1].errorbar(EPS,[r[key+'_mean'] for r in rs],yerr=[r[key+'_sample_std'] for r in rs],fmt=style,color=color,capsize=3,label=probe+' '+label)
    axes[0].axhline(1,color='grey');axes[0].set(title='Before/after class-component removal',xlabel='Fixed epsilon',ylabel='Domain / matched-random margin-drop ratio');axes[0].legend(fontsize=7)
    axes[1].set(title='Class-orthogonal attacks',xlabel='Fixed epsilon',ylabel='True logit-margin drop');axes[1].legend(fontsize=7)
    rows=[[r for r in s['energy_aggregate'] if r['probe']==probe and r['method']=='domain' and r['phase']==phase] for probe in PROBES for phase in ['before','class_orthogonal']]
    axes[2].bar(range(4),[100*r[0]['class_between_energy_fraction_mean'] if r else np.nan for r in rows],color=['#457b9d','#9bbacc','#e76f51','#edb0a1'])
    axes[2].set(title='Source class-between energy captured',ylabel='Energy (%)',xticks=range(4),xticklabels=['mean before','mean after','rank4 before','rank4 after']);axes[2].tick_params(axis='x',rotation=20)
    fig.suptitle('Frozen class-orthogonal scene-direction control; '+s['decision'])
    fig.savefig(OUT/'class_orthogonal_summary.png',dpi=180);plt.close(fig)
    print('\n'.join(lines[:21]),flush=True)

if __name__=='__main__':main()
