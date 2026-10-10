"""Offline GT evaluation, strictly after confidence/decile freeze."""
import json
import numpy as np
from common import HERE,OUT,DATA,SEEDS,KS,NAMES,sha,ahash,dump,csvwrite,signals,auc

def subsets(y,r0):
    out={'all_disagreement':np.ones(len(y),bool)}
    for c in range(7):out[f'C{c+1}']=y==c
    for c in range(7):
        for d in range(7):
            if c!=d and ((y==c)&(r0==d)).any():out[f'C{c+1}_to_C{d+1}']=(y==c)&(r0==d)
    return out

def distribution(x):
    if not len(x):return {'n':0,'mean':None,'std':None,**{key:None for key in ['min','q10','q25','q50','q75','q90','max']}}
    vals=np.quantile(x,[0,.1,.25,.5,.75,.9,1])
    return {'n':len(x),'mean':float(x.mean()),'std':float(x.std(ddof=1)) if len(x)>1 else None,
            **dict(zip(['min','q10','q25','q50','q75','q90','max'],map(float,vals)))}

def aggregate(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[key] for key in keys),[]).append(r)
    out=[]
    for group,items in groups.items():
        row=dict(zip(keys,group))
        for key in metrics:
            a=np.array([r[key] for r in items if r.get(key) is not None],float)
            row[key+'_valid_seeds']=len(a)
            row[key+'_mean']=float(a.mean()) if len(a) else None
            row[key+'_sample_std']=float(a.std(ddof=1)) if len(a)>1 else None
        out.append(row)
    return out

def main():
    assert auc([0,1],[0,1])==1 and auc([0,1],[1,0])==0 and auc([0,1],[1,1])==.5
    assert auc([1,1],[1,2]) is None
    gate=json.loads((OUT/'ALL_ROUTER_CONFIDENCE_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened']
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    print('All predictions/confidence/decile hashes verified. Opening GT offline.',flush=True)
    y=np.load(DATA/'target_labels.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    counts,aucs,dists,bins,agreement=[],[],[],[],[]
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';r0=np.load(folder/'R0_predictions.npy');r0right=r0==y
        subsets0=subsets(y,r0)
        for k in KS:
            a=np.load(folder/f'confidence_k{k}.npz');q=np.load(folder/f'disagreement_deciles_k{k}.npz')
            agreement_matrix=a['pairwise_agreement']
            for r in NAMES[1:]:
                pred=a[r+'__predictions'];right=pred==y;disagree=q[r+'__disagreement']
                assert np.array_equal(disagree,pred!=r0)
                benefit=~r0right&right;harm=r0right&~right;bothwrong=~r0right&~right
                assert not (disagree&r0right&right).any()
                assert np.array_equal((benefit|harm|bothwrong)&disagree,disagree)
                sig=signals(a,r)
                for sub,sm in subsets0.items():
                    mask=sm&disagree;n=int(mask.sum())
                    if not n:continue
                    nb=int((benefit&mask).sum());nh=int((harm&mask).sum());nw=int((bothwrong&mask).sum())
                    prefix={'seed':seed,'k':k,'alternative':r,'subset':sub}
                    counts.append({**prefix,'n_disagreement':n,'n_beneficial':nb,'n_harmful':nh,'n_both_wrong':nw,
                        'alternative_accuracy':nb/n,'R0_accuracy':nh/n,'both_wrong_fraction':nw/n,
                        'beneficial_fraction_single_correct':nb/(nb+nh) if nb+nh else None})
                    for name,(score,orientation) in sig.items():
                        sp={**prefix,'signal':name}
                        one=mask&(benefit|harm)
                        raw=auc(benefit[one],score[one]);oriented=auc(benefit[one],orientation*score[one])
                        au=auc(right[mask],score[mask]);r0au=auc(r0right[mask],score[mask])
                        aucs.append({**sp,'n_disagreement':n,'n_beneficial':nb,'n_harmful':nh,'n_both_wrong':nw,
                            'beneficial_vs_harmful_raw_AUROC':raw,'prespecified_benefit_orientation':orientation,
                            'beneficial_vs_harmful_oriented_AUROC':oriented,
                            'alternative_correct_raw_AUROC':au,'R0_correct_raw_AUROC':r0au,
                            'benefit_AUC_status':'defined' if raw is not None else 'undefined_missing_beneficial_or_harmful',
                            'alternative_AUC_status':'defined' if au is not None else 'undefined_single_outcome'})
                        for label,outcome in [('beneficial_override',benefit),('harmful_override',harm),('both_wrong',bothwrong)]:
                            dists.append({**sp,'outcome':label,**distribution(score[mask&outcome])})
                        edges=q[r+'__'+name+'__edges'];assign=q[r+'__'+name+'__bins']
                        for b in range(10):
                            bm=mask&(assign==b);bn=int(bm.sum());bnbe=int((bm&benefit).sum());bnha=int((bm&harm).sum())
                            bins.append({**sp,'bin':b+1,'edge_lower':float(edges[b-1]) if b else None,
                                'edge_upper':float(edges[b]) if b<9 else None,'n':bn,
                                'n_beneficial':bnbe,'n_harmful':bnha,'n_both_wrong':int((bm&bothwrong).sum()),
                                'alternative_accuracy':bnbe/bn if bn else None,'R0_accuracy':bnha/bn if bn else None,
                                'both_wrong_fraction':float((bm&bothwrong).sum()/bn) if bn else None,
                                'beneficial_fraction_single_correct':bnbe/(bnbe+bnha) if bnbe+bnha else None})
                    for j,nm in enumerate(NAMES):
                        for jj,nm2 in enumerate(NAMES):
                            if j<jj:agreement.append({**prefix,'router_A':nm,'router_B':nm2,'agreement_fraction':float(agreement_matrix[mask,j,jj].mean())})
            print(seed,'k',k,'all disagreement reliability scores completed',flush=True)
    csvwrite('disagreement_counts_per_seed.csv',counts)
    csvwrite('confidence_AUROC_per_seed.csv',aucs)
    csvwrite('confidence_distributions_per_seed.csv',dists)
    csvwrite('conditional_accuracy_deciles_per_seed.csv',bins)
    csvwrite('cross_stage_agreement_per_seed.csv',agreement)
    for name,rows in [('confidence_AUROC_per_class.csv',aucs),('confidence_distributions_per_class.csv',dists),('conditional_accuracy_per_class.csv',bins)]:
        csvwrite(name,[r for r in rows if r['subset'].startswith('C') and '_to_' not in r['subset']])
    for name,rows in [('confidence_AUROC_confusion_pairs.csv',aucs),('confidence_distributions_confusion_pairs.csv',dists),('conditional_accuracy_confusion_pairs.csv',bins)]:
        csvwrite(name,[r for r in rows if '_to_' in r['subset']])
    ac=aggregate(aucs,['k','alternative','subset','signal'],['beneficial_vs_harmful_raw_AUROC','beneficial_vs_harmful_oriented_AUROC','alternative_correct_raw_AUROC','R0_correct_raw_AUROC'])
    bc=aggregate(bins,['k','alternative','subset','signal','bin'],['alternative_accuracy','R0_accuracy','both_wrong_fraction','beneficial_fraction_single_correct','n'])
    dc=aggregate(dists,['k','alternative','subset','signal','outcome'],['mean','std','q10','q25','q50','q75','q90','n'])
    cc=aggregate(counts,['k','alternative','subset'],['alternative_accuracy','R0_accuracy','both_wrong_fraction','beneficial_fraction_single_correct','n_disagreement','n_beneficial','n_harmful','n_both_wrong'])
    csvwrite('confidence_AUROC_seed_aggregate.csv',ac);csvwrite('conditional_accuracy_seed_aggregate.csv',bc)
    csvwrite('confidence_distributions_seed_aggregate.csv',dc);csvwrite('disagreement_counts_seed_aggregate.csv',cc)
    report(ac,bc,cc)
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_predictions_confidence_deciles_checkpoints_unchanged':True,
       'target_GT_only_after_all_router_confidence_freeze':True,'no_training_tuning_followup':True,
       'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='SCORING_COMPLETE.json'}})

def report(ac,bc,cc):
    def fmt(row,key):
        m=row.get(key+'_mean');sd=row.get(key+'_sample_std');n=row.get(key+'_valid_seeds',0)
        return 'NA' if m is None else f'{m:.4f} +/- {sd:.4f} (n={n})' if sd is not None else f'{m:.4f} +/- NA (n={n})'
    native={'R1':['alternative_cosine_margin'],'R2':['alternative_vote_margin','alternative_neighbor_agreement'],
            'R3':['alternative_cosine_margin'],'R4':['alternative_vote_margin','alternative_neighbor_agreement'],
            'R5':['alternative_cosine_margin'],'R6':['alternative_vote_margin','alternative_neighbor_agreement']}
    lines=['Houston router reliability V1: 3 seeds2100/2101/2102, originalC epoch200.',
        'Only R0/alternative disagreements analyzed. All predictions/confidence/agreements/decile edges frozen before target GT.',
        'AUROC in [0,1], mean +/- sampleSD across seeds; quantiles/conditional rates use each frozen disagreement population.',
        'Raw AUROC is preserved; oriented benefit hypotheses fixed before GT (alternative high confidence, R0 low margin/high entropy).',
        'Both-wrong is a separate category. Beneficial-vs-harmful AUC excludes both-wrong; alternative-correct AUC retains both-wrong.',
        'No threshold/weights or combined selector is proposed or evaluated.','']
    for sub in ['all_disagreement','C1','C7','C1_to_C7','C7_to_C1','C7_to_C6']:
        lines.append('Subset '+sub)
        for k in KS:
            for r in NAMES[1:]:
                count=[x for x in cc if x['k']==k and x['alternative']==r and x['subset']==sub]
                if not count:continue
                lines.append(f'  k{k} {r}: disagreement n='+fmt(count[0],'n_disagreement')+'; beneficial rate='+fmt(count[0],'alternative_accuracy')+'; harmful rate='+fmt(count[0],'R0_accuracy')+'; both-wrong rate='+fmt(count[0],'both_wrong_fraction'))
                for name in native[r]+['R0_logit_margin','R0_entropy','alternative_agreement_other_stages','alternative_matched_stage_agreement']:
                    rows=[x for x in ac if x['k']==k and x['alternative']==r and x['subset']==sub and x['signal']==name]
                    if rows:lines.append('    '+name+': benefit oriented AUROC='+fmt(rows[0],'beneficial_vs_harmful_oriented_AUROC')+'; alternative-correct raw AUROC='+fmt(rows[0],'alternative_correct_raw_AUROC'))
    lines.extend(['','Fixed-decile conditional accuracy (all disagreements):'])
    for k in KS:
        for r in NAMES[1:]:
            for name in native[r]:
                rows=[x for x in bc if x['k']==k and x['alternative']==r and x['subset']=='all_disagreement' and x['signal']==name]
                lines.append(f'k{k} {r} {name}')
                for row in rows:lines.append(f"  bin{row['bin']}: alternative_accuracy="+fmt(row,'alternative_accuracy')+'; R0_accuracy='+fmt(row,'R0_accuracy')+'; both_wrong='+fmt(row,'both_wrong_fraction')+'; n='+fmt(row,'n'))
    lines.extend(['','Limits: binary/discrete confidence ties yield empty decile bins, not artificially split samples.',
        'GT class/pair subsets use full-disagreement frozen edges. Missing outcome groups produce undefined AUROC.',
        'C7->C6 cannot have harmful overrides (R0 is wrong by definition); use alternative-correct AUROC there.',
        'No AUROC orientation is chosen after scoring. Raw confidence scales differ across models/routers; do not pool them.',
        'All stage votes are correlated; agreement is an observed signal, not independent evidence.',
        'One scene pair,3 optimization seeds and overlapping patches limit inference. No significance, cross-scene or operational-threshold claim.'])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    # Keep findings focused on prespecified signals rather than GT-based best-signal selection.
    brief=lines[:6]+['','All-disagreement native confidence:']
    for k in KS:
        for r in NAMES[1:]:
            for name in native[r]:
                row=next(x for x in ac if x['k']==k and x['alternative']==r and x['subset']=='all_disagreement' and x['signal']==name)
                brief.append(f'k{k} {r} {name}: benefit oriented AUROC='+fmt(row,'beneficial_vs_harmful_oriented_AUROC')+'; alternative-correct AUROC='+fmt(row,'alternative_correct_raw_AUROC'))
    brief.extend(['','Full C1/C7/confusion analyses and frozen-decile conditional accuracy in REPORT/CSV.',
        'AUROC alone cannot establish a safe rule for overriding R0. No parameters were selected and no follow-up was started.'])
    (OUT/'FINDINGS.txt').write_text('\n'.join(brief)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for offset,k,color in [(-.17,5,'#457b9d'),(.17,15,'#e9c46a')]:
        rows=[next(x for x in ac if x['k']==k and x['alternative']==r and x['subset']=='all_disagreement' and x['signal']==native[r][0]) for r in NAMES[1:]]
        axes[0].bar(np.arange(6)+offset,[x['beneficial_vs_harmful_oriented_AUROC_mean'] for x in rows],.34,
            yerr=[x['beneficial_vs_harmful_oriented_AUROC_sample_std'] or 0 for x in rows],capsize=3,label=f'k{k}',color=color)
    axes[0].axhline(.5,color='grey',linestyle='--');axes[0].set(xticks=range(6),xticklabels=NAMES[1:],ylim=(0,1),ylabel='AUROC',title='Native margin: beneficial vs harmful override');axes[0].legend()
    # Prespecified T kNN, both k; discrete tied bins can be empty.
    for k,color in [(5,'#457b9d'),(15,'#e9c46a')]:
        rows=[x for x in bc if x['k']==k and x['alternative']=='R4' and x['subset']=='all_disagreement' and x['signal']=='alternative_vote_margin' and x['alternative_accuracy_mean'] is not None]
        axes[1].plot([x['bin'] for x in rows],[100*x['alternative_accuracy_mean'] for x in rows],'o-',color=color,label=f'T kNN k{k}')
        axes[1].plot([x['bin'] for x in rows],[100*x['R0_accuracy_mean'] for x in rows],'x--',color=color,label=f'R0 on same k{k} bins')
    axes[1].set(xlabel='Frozen disagreement-confidence decile bin',ylabel='Conditional accuracy (%)',ylim=(0,100),title='T kNN vote margin on disagreements');axes[1].legend()
    fig.suptitle('Frozen router reliability; 3seed mean +/- SD; no threshold selection')
    fig.savefig(OUT/'router_reliability_summary.png',dpi=180);plt.close(fig)
    print('\n'.join(brief),flush=True)

if __name__=='__main__':main()
