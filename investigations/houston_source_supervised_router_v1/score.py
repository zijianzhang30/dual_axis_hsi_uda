"""Target GT scoring only after every source-trained decision and transport is frozen."""
import json
import numpy as np
from common import OUT,DATA,SEEDS,KS,CONFIG,sha,ahash,dump,csvwrite

def metrics(y,p):
    cm=np.bincount(y*7+p,minlength=49).reshape(7,7);n=cm.sum(1);correct=np.diag(cm)
    recall=np.divide(correct,n,out=np.zeros(7,float),where=n>0)
    po=float(correct.sum()/len(y));pe=float(cm.sum(0)@cm.sum(1)/len(y)**2)
    return {'OA':100*po,'AA':100*float(recall[n>0].mean()),'Kappa':(po-pe)/(1-pe) if pe<1 else None,
        'class_recall':(100*recall).tolist(),'confusion':cm.tolist()}

def subset_masks(y,r0):
    right=y==r0;out={'all':np.ones(len(y),bool),'R0_correct':right,'R0_error':~right}
    for c in range(7):
        out[f'C{c+1}']=y==c;out[f'C{c+1}_R0_correct']=(y==c)&right;out[f'C{c+1}_R0_error']=(y==c)&~right
        for j in range(7):
            if j!=c and ((y==c)&(r0==j)).any():out[f'C{c+1}_to_C{j+1}']=(y==c)&(r0==j)
    return out

def aggregate(rows,groups,values):
    out=[];d={}
    for r in rows:d.setdefault(tuple(r[g] for g in groups),[]).append(r)
    for group,rs in d.items():
        row=dict(zip(groups,group))
        for key in values:
            a=np.array([r[key] for r in rs if r[key] is not None],float)
            row[key+'_mean']=float(a.mean()) if len(a) else None
            row[key+'_sample_std']=float(a.std(ddof=1)) if len(a)>1 else None
            row[key+'_valid_seeds']=len(a)
        out.append(row)
    return out

def main():
    gate=json.loads((OUT/'ALL_TARGET_DECISIONS_FROZEN.json').read_text());assert gate['all_seeds_completed'] and not gate['target_GT_opened']
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    print('All source-trained target decisions/transport hashes verified. Reading target GT for scoring only.',flush=True)
    y=np.load(DATA/'target_labels.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    rows,subs,oracles,gate_counts=[],[],[],[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';r0=np.load(f/'target_predictions.npy');correct=r0==y;original=metrics(y,r0);masks=subset_masks(y,r0)
        for k in KS:
            inp=np.load(f/f'target_gate_inputs_k{k}.npz');allp=inp['router_predictions'];available=(allp==y[:,None]).any(1)
            oraclepred=np.where(available,y,r0);om=metrics(y,oraclepred)
            oracles.append({'seed':seed,'k':k,'routing_OA_ceiling':om['OA'],'routing_AA_ceiling':om['AA'],'routing_Kappa_ceiling':om['Kappa'],
                'R0_error_available_percent':float(100*available[~correct].mean()),'R0_error_available_n':int((available&~correct).sum())})
            np.savez_compressed(f/f'oracle_selector_k{k}_GT_only.npz',any_router_correct=available,oracle_predictions=oraclepred)
            for method in ['G0','D00_R0','G1','G2','G3']:
                a=np.load(f/f'{method}_k{k}_readouts.npz');pred=a['predictions'];route=a['routing'];changed=route!=r0
                selector=a['selected_router'];override=selector>0 if method in ['G2','G3'] else changed
                m=metrics(y,pred);rm=metrics(y,route)
                recovery=(~correct)&(pred==y);damage=correct&(pred!=y)
                beneficial=(~correct)&(route==y);harmful=correct&(route!=y)
                rows.append({'seed':seed,'k':k,'method':method,**{key:m[key] for key in ['OA','AA','Kappa']},
                    **{f'delta_{key}_vs_originalC':m[key]-original[key] for key in ['OA','AA','Kappa']},
                    'routing_OA':rm['OA'],'routing_AA':rm['AA'],'routing_Kappa':rm['Kappa'],
                    'error_recovery_percent':float(100*recovery.sum()/(~correct).sum()),
                    'correct_damage_percent':float(100*damage.sum()/correct.sum()),
                    'keep_percent':float(100*(~override).mean()),'override_percent':float(100*override.mean()),
                    'changed_routing_percent':float(100*changed.mean()),'final_prediction_changed_percent':float(100*(pred!=r0).mean()),
                    'n_recovered':int(recovery.sum()),'n_damaged':int(damage.sum()),
                    'net_correct_count':int(recovery.sum()-damage.sum()),
                    'n_changed_route_recovered':int((recovery&changed).sum()),'n_unchanged_route_recovered':int((recovery&~changed).sum()),
                    'n_changed_route_damaged':int((damage&changed).sum()),'n_unchanged_route_damaged':int((damage&~changed).sum())})
                assert int(recovery.sum()-damage.sum())==int((pred==y).sum()-correct.sum())
                for sub,mask in masks.items():
                    n=int(mask.sum())
                    if not n:continue
                    er=int((mask&~correct).sum());cr=int((mask&correct).sum());ov=int((mask&override).sum())
                    subs.append({'seed':seed,'k':k,'method':method,'subset':sub,'n':n,
                        'classifier_accuracy_percent':float(100*(pred[mask]==y[mask]).mean()),
                        'original_accuracy_percent':float(100*correct[mask].mean()),
                        'accuracy_delta_pp':float(100*((pred[mask]==y[mask]).mean()-correct[mask].mean())),
                        'routing_accuracy_percent':float(100*(route[mask]==y[mask]).mean()),
                        'override_percent':float(100*override[mask].mean()),'keep_percent':float(100*(~override[mask]).mean()),
                        'changed_routing_percent':float(100*changed[mask].mean()),
                        'n_recovered':int((mask&recovery).sum()),'n_damaged':int((mask&damage).sum()),
                        'error_recovery_percent':float(100*(mask&recovery).sum()/er) if er else None,
                        'correct_damage_percent':float(100*(mask&damage).sum()/cr) if cr else None,
                        'n_beneficial_route_override':int((mask&beneficial&override).sum()),
                        'n_harmful_route_override':int((mask&harmful&override).sum()),
                        'n_override_both_wrong':int((mask&override&~correct&(route!=y)).sum()),
                        'override_final_accuracy_percent':float(100*(pred[mask&override]==y[mask&override]).mean()) if ov else None})
                if method=='G3':
                    for j in range(7):gate_counts.append({'seed':seed,'k':k,'selected_router':f'R{j}',
                         'n':int((selector==j).sum()),'percent':float(100*(selector==j).mean())})
                print(seed,k,method,'OA/AA',m['OA'],m['AA'],'delta OA',m['OA']-original['OA'],'override%',100*override.mean(),flush=True)
    csvwrite('target_metrics_per_seed.csv',rows);csvwrite('target_subset_statistics.csv',subs)
    csvwrite('target_per_class.csv',[r for r in subs if r['subset'].startswith('C') and '_to_' not in r['subset']])
    csvwrite('target_confusion_pairs.csv',[r for r in subs if '_to_' in r['subset']])
    csvwrite('oracle_selector_availability.csv',oracles);csvwrite('selected_router_counts.csv',gate_counts)
    keys=[key for key in rows[0] if key not in ['seed','k','method']]
    agg=aggregate(rows,['k','method'],keys);csvwrite('target_metrics_seed_aggregate.csv',agg)
    skeys=[key for key in subs[0] if key not in ['seed','k','method','subset']]
    sagg=aggregate(subs,['k','method','subset'],skeys);csvwrite('target_subset_seed_aggregate.csv',sagg)
    go=all(r['delta_OA_vs_originalC']>0 for r in rows if r['method']=='G3')
    assert len([r for r in rows if r['method']=='G3'])==6
    summary={'fixed_config':CONFIG,'target_metrics':rows,'seed_aggregate':agg,'subset_aggregate':sagg,'oracle_selector':oracles,
             'decision':'Go' if go else 'No-Go','Go_criterion':CONFIG['Go_rule'],
             'no_automatic_followup':True,'limitations':['Only one Houston scene pair,3 model seeds; not independent scenes.',
              'Source perturbations may not reproduce target class-conditional shifts.',
              'Original source spatial query overlap excluded; source reference prototype information still shared across selector training samples.',
              'G1 votes all six alternatives in this prespecified controlled run; differs from earlier R1-R4 majority.',
              'G2 source ECDF percentiles compare confidence ranks, not calibrated correctness probabilities.',
              'Oracle selector is GT-only routing signal availability, not a transformed-classifier upper bound.']}
    dump(OUT/'summary.json',summary);report(summary)
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_main_checkpoints_source_selectors_target_decisions_statistics_unchanged':True,
       'target_GT_only_after_all_seeds_all_k_freeze':True,'no_automatic_followup':True,'decision':summary['decision'],
       'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='SCORING_COMPLETE.json'}})

def report(s):
    def fmt(r,key):
        m=r.get(key+'_mean');sd=r.get(key+'_sample_std')
        if m is None:return 'NA'
        return f'{m:.5f} +/- {sd:.5f}' if sd is not None else f'{m:.5f} +/- NA'
    lines=['Houston source-supervised router V1: '+s['decision'],
        'Original C seeds2100/2101/2102 epoch200; main CNN/BN/tokenizer/Transformer/classifier frozen.',
        'Source GT only:808-parameter two-stage MLP,hidden16,last epoch60; no early stop/search/target supervision.',
        '4 source views with fixed lightweight affine/intensity/noise; source2393 train and127 fixed heldout origins.',
        'Source query/overlapping patches excluded from centroid/kNN references; target original R0-R6 exactly replayed.',
        'Both k5/15 retained; G1 six-alternative majority; G2 highest source-native confidence ECDF percentile.',
        'All target gate inputs/routing/D00 transport/final predictions frozen+hashed before target GT.',
        'Go criterion fixed: '+s['Go_criterion'],'']
    for k in KS:
        lines.append(f'k{k},mean +/- sampleSD across3seeds:')
        for r in s['seed_aggregate']:
            if r['k']!=k:continue
            lines.append(r['method']+': OA='+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; Kappa='+fmt(r,'Kappa')+
              '; paired OA delta='+fmt(r,'delta_OA_vs_originalC')+'pp; AA delta='+fmt(r,'delta_AA_vs_originalC')+'pp; Kappa delta='+fmt(r,'delta_Kappa_vs_originalC'))
            lines.append('  error recovery='+fmt(r,'error_recovery_percent')+'%; correct damage='+fmt(r,'correct_damage_percent')+'%; gate keep='+fmt(r,'keep_percent')+'%; override='+fmt(r,'override_percent')+'%; class routing changed='+fmt(r,'changed_routing_percent')+'%; routing OA='+fmt(r,'routing_OA')+'%')
        for r in s['target_metrics']:
            if r['k']==k and r['method']=='G3':lines.append(f"  G3 seed{r['seed']}: OA delta={r['delta_OA_vs_originalC']:.5f}pp; recovered={r['n_recovered']}; damaged={r['n_damaged']}; net={r['net_correct_count']}; unchanged-route recovered/damaged={r['n_unchanged_route_recovered']}/{r['n_unchanged_route_damaged']}")
    lines.extend(['','Focused classifier/override outcomes:'])
    for sub in ['C1','C7','C1_R0_error','C7_R0_error','C7_to_C6']:
        for r in s['subset_aggregate']:
            if r['subset']==sub and r['method']=='G3':lines.append(f"G3 k{r['k']} {sub}: classifier accuracy="+fmt(r,'classifier_accuracy_percent')+'%; delta='+fmt(r,'accuracy_delta_pp')+'pp; recovery='+fmt(r,'error_recovery_percent')+'%; damage='+fmt(r,'correct_damage_percent')+'%; override='+fmt(r,'override_percent')+'%')
    lines.extend(['','Oracle selector availability (only scoring, not a deployable rule):'])
    for k in KS:
        rs=[r for r in s['oracle_selector'] if r['k']==k]
        for key in ['routing_OA_ceiling','routing_AA_ceiling','R0_error_available_percent']:
            a=np.array([r[key] for r in rs]);lines.append(f'k{k} {key}: {a.mean():.5f} +/- {a.std(ddof=1):.5f}%')
    lines.extend(['','Limits:',*s['limitations'],'','No automatic additional training/complex gate.'])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    methods=['G0','D00_R0','G1','G2','G3']
    for offset,k,color in [(-.18,5,'#457b9d'),(.18,15,'#e9c46a')]:
        rs=[next(r for r in s['seed_aggregate'] if r['k']==k and r['method']==m) for m in methods]
        axes[0].bar(np.arange(5)+offset,[r['delta_OA_vs_originalC_mean'] for r in rs],.36,yerr=[r['delta_OA_vs_originalC_sample_std'] for r in rs],capsize=3,color=color,label=f'k{k}')
        axes[1].bar(np.arange(5)+offset,[r['override_percent_mean'] for r in rs],.36,yerr=[r['override_percent_sample_std'] for r in rs],capsize=3,color=color,label=f'k{k}')
    axes[0].axhline(0,color='grey');axes[0].set(title='Final classifier OA change',ylabel='vs original C (pp)',xticks=range(5),xticklabels=methods);axes[0].legend()
    axes[1].set(title='Override decisions',ylabel='Target samples (%)',xticks=range(5),xticklabels=methods);axes[1].legend()
    for offset,key,color,label in [(-.18,'error_recovery_percent','#2a9d8f','Original errors recovered'),(.18,'correct_damage_percent','#e76f51','Original correct damaged')]:
        rs=[next(r for r in s['seed_aggregate'] if r['k']==k and r['method']=='G3') for k in KS]
        axes[2].bar(np.arange(2)+offset,[r[key+'_mean'] for r in rs],.36,yerr=[r[key+'_sample_std'] for r in rs],capsize=3,color=color,label=label)
    axes[2].set(title='Learned selector recovery/damage',ylabel='Original subset (%)',xticks=[0,1],xticklabels=['k5','k15']);axes[2].legend()
    fig.suptitle('Source-only tiny reliability selector; fixed original C; '+s['decision'])
    fig.savefig(OUT/'source_supervised_router_summary.png',dpi=180);plt.close(fig)
    print('\n'.join(lines[:32]),flush=True)

if __name__=='__main__':main()
