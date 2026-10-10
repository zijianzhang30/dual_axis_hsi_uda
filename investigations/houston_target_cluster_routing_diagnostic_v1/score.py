"""GT scores and contextual reference comparisons after all construction freezes."""
import json
import numpy as np
from common import OUT,CACHE,FACT,LEARNED,DATA,SEEDS,STAGES,CONFIG,sha,ahash,dump,csvwrite,metric,subsets,aggregate

def authenticated(root,rel,gate):
    assert sha(root/rel)==gate['files_sha256'][rel]
    return np.load(root/rel)

def main():
    gate=json.loads((OUT/'ALL_CLUSTER_ROUTING_TRANSPORT_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened']
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    print('All cluster/matching/routing/transport hashes verified. Reading target GT offline.',flush=True)
    y=np.load(DATA/'target_labels.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    # Oracle references are first opened here, after all current decisions are frozen.
    og=json.loads((FACT/'ALL_ORACLE_RESULTS_FROZEN.json').read_text())
    rg=json.loads((CACHE/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text())
    lg=json.loads((LEARNED/'ALL_TARGET_DECISIONS_FROZEN.json').read_text())
    lp=json.loads((LEARNED/'provenance.json').read_text())
    cluster_rows,purity_rows,confusion_rows,matching_rows,rows,subset_rows,refs,fragment_rows=[],[],[],[],[],[],[],[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';r0=np.load(f/'target_predictions.npy');correct=r0==y;masks=subsets(y,r0)
        methods={'R0':np.load(f/'R0_readouts.npz'),'D00':np.load(f/'D00_readouts.npz')}
        for oracle in ['D01','D11']:
            rel=f'seed_{seed}/{oracle}_readouts.npz';a=authenticated(FACT,rel,og)
            np.savez_compressed(f/f'{oracle}_oracle_reference_readouts.npz',logits=a['logits'],predictions=a['predictions'])
            methods[oracle]=a
        d00=metric(y,methods['D00']['predictions']);orig=metric(y,r0)
        oracle_metrics={name:metric(y,methods[name]['predictions']) for name in ['D01','D11']}
        for stage in STAGES:
            c=np.load(f/f'{stage}_clustering_matching.npz');ids=c['clusters'];mapping=c['cluster_to_source_class']
            cm=np.bincount(7*ids+y,minlength=49).reshape(7,7);size=cm.sum(1);dominant=cm.argmax(1);purity=cm.max(1)/size
            weighted=float(100*cm.max(1).sum()/len(y));macro=float(100*purity.mean())
            purity_rows.append({'seed':seed,'stage':stage,'weighted_purity_percent':weighted,'macro_purity_percent':macro,
                'n_iter':int(c['n_iter']),'inertia':float(c['inertia']),
                'n_unique_majority_GT_classes':int(len(np.unique(dominant))),
                'n_clusters_majority_C6':int((dominant==5).sum()),
                'n_clusters_matched_source_equals_majority_GT':int((mapping==dominant).sum())})
            for j in range(7):
                cluster_rows.append({'seed':seed,'stage':stage,'cluster':j,'n':int(size[j]),
                    'purity_percent':float(100*purity[j]),'GT_majority_class':int(dominant[j]+1),
                    'matched_source_class':int(mapping[j]+1),
                    'matched_class_correct_fraction':float(cm[j,mapping[j]]/size[j]),
                    'matched_class_cosine':float(1-c['cosine_cost'][j,mapping[j]]),
                    'assigned_class_equals_GT_majority':bool(mapping[j]==dominant[j])})
                for cl in range(7):
                    confusion_rows.append({'seed':seed,'stage':stage,'cluster':j,'matched_source_class':int(mapping[j]+1),
                       'true_class':cl+1,'n':int(cm[j,cl]),'row_fraction':float(cm[j,cl]/size[j])})
                    matching_rows.append({'seed':seed,'stage':stage,'cluster':j,'source_class':cl+1,
                        'cosine':float(1-c['cosine_cost'][j,cl]),'cosine_cost':float(c['cosine_cost'][j,cl]),'matched':bool(mapping[j]==cl)})
            for cl in range(7):
                percluster=cm[:,cl];n=int(percluster.sum())
                fragment_rows.append({'seed':seed,'stage':stage,'true_class':cl+1,'n':n,
                    'class_in_largest_single_cluster_percent':float(100*percluster.max()/n),
                    'class_in_its_Hungarian_assigned_cluster_percent':float(100*percluster[np.flatnonzero(mapping==cl)[0]]/n),
                    'n_clusters_containing_class':int((percluster>0).sum())})
            np.save(f/f'{stage}_cluster_class_confusion_GT.npy',cm)
            methods[stage+'_cluster']=np.load(f/f'{stage}_cluster_readouts.npz')
            print(seed,stage,'GT weighted/macro purity',weighted,macro,'majority classes',(dominant+1).tolist(),'assigned',(mapping+1).tolist(),flush=True)
        for method,a in methods.items():
            p=a['predictions'];routing=y if method in ['D01','D11'] else a['routing']
            m=metric(y,p);rm=metric(y,routing);recover=(~correct)&(p==y);damage=correct&(p!=y)
            routechanged=routing!=r0
            row={'seed':seed,'method':method,**{key:m[key] for key in ['OA','AA','Kappa']},
                **{key+'_delta_vs_D00':m[key]-d00[key] for key in ['OA','AA','Kappa']},
                **{key+'_delta_vs_R0':m[key]-orig[key] for key in ['OA','AA','Kappa']},
                'routing_OA':rm['OA'],'routing_AA':rm['AA'],'routing_Kappa':rm['Kappa'],
                'routing_correct_on_R0_error_percent':float(100*(routing[~correct]==y[~correct]).mean()),
                'routing_correct_on_R0_correct_percent':float(100*(routing[correct]==y[correct]).mean()),
                'route_changed_percent':float(100*routechanged.mean()),
                'original_error_recovery_percent':float(100*recover.sum()/(~correct).sum()),
                'original_correct_damage_percent':float(100*damage.sum()/correct.sum()),
                'n_recovered':int(recover.sum()),'n_damaged':int(damage.sum()),'net_correct_count':int(recover.sum()-damage.sum())}
            for oracle in ['D01','D11']:
                gap=oracle_metrics[oracle]['OA']-d00['OA']
                row[f'OA_gap_closure_vs_{oracle}_percent']=100*(m['OA']-d00['OA'])/gap if gap>0 else None
            rows.append(row)
            assert int((p==y).sum()-correct.sum())==row['net_correct_count']
            for sub,mask in masks.items():
                n=int(mask.sum())
                if not n:continue
                er=int((mask&~correct).sum());cr=int((mask&correct).sum())
                subset_rows.append({'seed':seed,'method':method,'subset':sub,'n':n,
                    'routing_accuracy_percent':float(100*(routing[mask]==y[mask]).mean()),
                    'classifier_accuracy_percent':float(100*(p[mask]==y[mask]).mean()),
                    'classifier_delta_vs_D00_pp':float(100*((p[mask]==y[mask]).mean()-(methods['D00']['predictions'][mask]==y[mask]).mean())),
                    'route_changed_percent':float(100*routechanged[mask].mean()),
                    'n_recovered':int((recover&mask).sum()),'n_damaged':int((damage&mask).sum()),
                    'error_recovery_percent':float(100*(recover&mask).sum()/er) if er else None,
                    'correct_damage_percent':float(100*(damage&mask).sum()/cr) if cr else None})
            refs.append({'seed':seed,'reference':method,'routing_OA':rm['OA'],'routing_AA':rm['AA'],
                'routing_R0_error_correct_percent':row['routing_correct_on_R0_error_percent'],
                'classifier_OA':m['OA'],'classifier_AA':m['AA'],
                'classifier_OA_delta_vs_D00':m['OA']-d00['OA']})
            print(seed,method,'routing OA',rm['OA'],'classifier OA',m['OA'],'delta vs D00',m['OA']-d00['OA'],flush=True)
        assert lp['seeds'][str(seed)]['checkpoint_sha256']==prov['seeds'][str(seed)]['checkpoint_sha256']
        learnedcoords=authenticated(LEARNED,f'seed_{seed}/target_coords.npy',lg)
        assert np.array_equal(learnedcoords,np.load(f/'target_coords.npy'))
        for k in [5,15]:
            for stage,r in [('T','R4'),('Z','R2')]:
                routing=authenticated(CACHE,f'seed_{seed}/{r}_k{k}_predictions.npy',rg)
                rm=metric(y,routing)
                refs.append({'seed':seed,'reference':f'{stage}_source_kNN_k{k}','routing_OA':rm['OA'],'routing_AA':rm['AA'],
                    'routing_R0_error_correct_percent':float(100*(routing[~correct]==y[~correct]).mean()),
                    'classifier_OA':None,'classifier_AA':None,'classifier_OA_delta_vs_D00':None})
            a=authenticated(LEARNED,f'seed_{seed}/G3_k{k}_readouts.npz',lg)
            rm=metric(y,a['routing']);m=metric(y,a['predictions'])
            refs.append({'seed':seed,'reference':f'G3_source_supervised_k{k}','routing_OA':rm['OA'],'routing_AA':rm['AA'],
                'routing_R0_error_correct_percent':float(100*(a['routing'][~correct]==y[~correct]).mean()),
                'classifier_OA':m['OA'],'classifier_AA':m['AA'],'classifier_OA_delta_vs_D00':m['OA']-d00['OA']})
    csvwrite('cluster_per_seed.csv',cluster_rows);csvwrite('cluster_purity_per_seed.csv',purity_rows)
    csvwrite('cluster_class_confusion.csv',confusion_rows);csvwrite('source_cluster_cosine_matching.csv',matching_rows)
    csvwrite('class_cluster_fragmentation.csv',fragment_rows);csvwrite('metrics_per_seed.csv',rows)
    csvwrite('subset_statistics.csv',subset_rows)
    csvwrite('per_class.csv',[r for r in subset_rows if r['subset'].startswith('C') and '_to_' not in r['subset']])
    csvwrite('confusion_pairs.csv',[r for r in subset_rows if '_to_' in r['subset']])
    csvwrite('routing_reference_comparison_per_seed.csv',refs)
    ma=aggregate(rows,['method'],[k for k in rows[0] if k not in ['seed','method']])
    pa=aggregate(purity_rows,['stage'],[k for k in purity_rows[0] if k not in ['seed','stage']])
    sa=aggregate(subset_rows,['method','subset'],[k for k in subset_rows[0] if k not in ['seed','method','subset']])
    ra=aggregate(refs,['reference'],[k for k in refs[0] if k not in ['seed','reference']])
    csvwrite('metrics_seed_aggregate.csv',ma);csvwrite('purity_seed_aggregate.csv',pa)
    csvwrite('subset_seed_aggregate.csv',sa);csvwrite('routing_reference_comparison_seed_aggregate.csv',ra)
    stable={stage:all(r['OA_delta_vs_D00']>0 for r in rows if r['method']==stage+'_cluster') for stage in STAGES}
    summary={'fixed_config':CONFIG,'metrics':rows,'metrics_aggregate':ma,'purity_aggregate':pa,
        'subsets_aggregate':sa,'references_aggregate':ra,'stage_stable_positive':stable,
        'decision':'Go' if any(stable.values()) else 'No-Go','no_automatic_followup':True,
        'limits':['One scene pair,3 optimization seeds; not independent scenes.',
            'Standard KMeans geometry on normalized flattened descriptors is a fixed hypothesis, not a general test of all clustering.',
            'Size-weighted purity and macro cluster purity are distinct from source-class routing accuracy.',
            'Cluster labels arbitrary across seeds; raw rows are not averaged.',
            'One-to-one matching requires one cluster per source class even when actual target classes fragment or merge.',
            'D01/D11 are GT-assisted transport references, not guaranteed mathematical bounds for every transformation.',
            'No negative gap-closure ratios are clipped; no best stage/K/start/method is selected.']}
    dump(OUT/'summary.json',summary);report(summary)
    for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_features_clustering_matching_routing_transport_statistics_checkpoints_unchanged':True,
        'target_GT_only_after_all_seeds_all_stages_freeze':True,'oracle_references_first_accessed_after_freeze':True,
        'decision':summary['decision'],'no_automatic_followup':True,
        'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='SCORING_COMPLETE.json'}})

def report(s):
    def fmt(r,key):
        m=r.get(key+'_mean');sd=r.get(key+'_sample_std')
        return 'NA' if m is None else f'{m:.5f} +/- {sd:.5f}' if sd is not None else f'{m:.5f} +/- NA'
    lines=['Houston target cluster routing V1: '+s['decision'],
        'Original Joint BN+C seeds2100/2101/2102 epoch200; no network training/backprop/tuning/follow-up.',
        'Target KMeans7,initkmeans++,n_init10,max_iter300,tol1e-4,lloyd,random_state0; all fixed.',
        'Sample L2-normalized T256/Z64; source2393 training references; Hungarian1-cosine matching.',
        'All current inputs/clustering/matching/routing/transport frozen and hash-verified before target GT.',
        'RawZ transport uses unchanged D00 pseudo statistics even for T-based routes.',
        'D01/D11 oracle references and previous kNN/G3 are only accessed for postfreeze comparison.',
        '3seed mean +/- sampleSD; gap closure is signed; negative means widening the oracle gap.','']
    for r in s['purity_aggregate']:
        lines.append(r['stage']+' purity: weighted='+fmt(r,'weighted_purity_percent')+'%; macro='+fmt(r,'macro_purity_percent')+
            '%; unique majority GT classes='+fmt(r,'n_unique_majority_GT_classes')+'; majorityC6 clusters='+fmt(r,'n_clusters_majority_C6'))
    for r in s['metrics_aggregate']:
        lines.append(r['method']+': routing OA='+fmt(r,'routing_OA')+'%; AA='+fmt(r,'routing_AA')+'%; correct on R0 errors='+fmt(r,'routing_correct_on_R0_error_percent')+'%')
        lines.append('  classifier OA='+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; Kappa='+fmt(r,'Kappa')+
            '; same-seed delta vsD00 OA='+fmt(r,'OA_delta_vs_D00')+'pp; AA='+fmt(r,'AA_delta_vs_D00')+'pp; Kappa='+fmt(r,'Kappa_delta_vs_D00'))
        lines.append('  original error recovery='+fmt(r,'original_error_recovery_percent')+'%; original correct damage='+fmt(r,'original_correct_damage_percent')+'%; recovered='+fmt(r,'n_recovered')+'; damaged='+fmt(r,'n_damaged'))
        if r['method'] in ['T_cluster','Z_cluster']:lines.append('  OA oracle gap closure D01='+fmt(r,'OA_gap_closure_vs_D01_percent')+'%; D11='+fmt(r,'OA_gap_closure_vs_D11_percent')+'%')
    lines.extend(['','Same-seed cluster transport OA changes vs D00:'])
    for r in s['metrics']:
        if r['method'] in ['T_cluster','Z_cluster']:lines.append(f"seed{r['seed']} {r['method']}={r['OA_delta_vs_D00']:.5f}pp")
    lines.extend(['','Focused routing and transport outcomes:'])
    for sub in ['C1_all','C7_all','C1_R0_error','C7_R0_error','C7_to_C6']:
        for r in s['subsets_aggregate']:
            if r['subset']==sub and r['method'] in ['T_cluster','Z_cluster']:
                lines.append(r['method']+' '+sub+': routing accuracy='+fmt(r,'routing_accuracy_percent')+'%; classifier accuracy='+fmt(r,'classifier_accuracy_percent')+'%; delta vsD00='+fmt(r,'classifier_delta_vs_D00_pp')+'pp; error recovery='+fmt(r,'error_recovery_percent')+'%; correct damage='+fmt(r,'correct_damage_percent')+'%')
    lines.extend(['','Frozen reference comparisons:'])
    for r in s['references_aggregate']:lines.append(r['reference']+': routing OA='+fmt(r,'routing_OA')+'%; AA='+fmt(r,'routing_AA')+'%; R0 error correct='+fmt(r,'routing_R0_error_correct_percent')+'%; classifier OA='+fmt(r,'classifier_OA')+'%; classifier delta vsD00='+fmt(r,'classifier_OA_delta_vs_D00')+'pp')
    lines.extend(['','Stable-positive criterion: '+CONFIG['Go_rule'],'Stage flags: '+str(s['stage_stable_positive']),
        'No automatic GMM/spectral/differentK or any additional variant.','Limits:',*s['limits']])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for i,stage in enumerate(STAGES):
        for j,seed in enumerate(SEEDS):
            cm=np.load(OUT/f'seed_{seed}/{stage}_cluster_class_confusion_GT.npy');cl=np.load(OUT/f'seed_{seed}/{stage}_clustering_matching.npz')
            frac=100*cm/cm.sum(1,keepdims=True);ax=axes[i,j]
            im=ax.imshow(frac,vmin=0,vmax=100,cmap='Blues')
            for row in range(7):
                for col in range(7):
                    if frac[row,col]>=1:ax.text(col,row,f'{frac[row,col]:.0f}',ha='center',va='center',fontsize=8,color='white' if frac[row,col]>60 else 'black')
            ax.set(title=f'{stage}, seed{seed}',xticks=range(7),xticklabels=[f'C{k+1}' for k in range(7)],
                yticks=range(7),yticklabels=[f"cl{k}->C{cl['cluster_to_source_class'][k]+1} (n={cm[k].sum()})" for k in range(7)],xlabel='True target class')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='GT composition within cluster (%)',shrink=.7)
    fig.suptitle('Frozen KMeans cluster composition; Hungarian source-class assignment; GT only after freeze')
    fig.savefig(OUT/'cluster_class_confusions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    names=['R0','D00','T_cluster','Z_cluster','D01','D11'];rs=[next(r for r in s['metrics_aggregate'] if r['method']==name) for name in names]
    axes[0].bar(range(6),[r['OA_mean'] for r in rs],yerr=[r['OA_sample_std'] for r in rs],capsize=3,color=['grey','grey','#457b9d','#e9c46a','#9ec5ab','#6b9f7f'])
    axes[0].set(xticks=range(6),xticklabels=names,ylabel='Frozen classifier OA (%)',ylim=(0,100),title='Unchanged D00-statistics transport')
    purity=[r for r in s['purity_aggregate']]
    for offset,key,color,label in [(-.2,'weighted_purity_percent','#457b9d','Weighted cluster purity'),(0,'macro_purity_percent','#e9c46a','Macro cluster purity')]:
        axes[1].bar(np.arange(2)+offset,[next(r for r in purity if r['stage']==stage)[key+'_mean'] for stage in STAGES],.2,
            yerr=[next(r for r in purity if r['stage']==stage)[key+'_sample_std'] for stage in STAGES],capsize=3,label=label,color=color)
    cluster=[next(r for r in rs if r['method']==stage+'_cluster') for stage in STAGES]
    axes[1].bar(np.arange(2)+.2,[r['routing_OA_mean'] for r in cluster],.2,yerr=[r['routing_OA_sample_std'] for r in cluster],capsize=3,label='Hungarian routing OA',color='#e76f51')
    axes[1].set(xticks=[0,1],xticklabels=STAGES,ylabel='Target samples (%)',ylim=(0,100),title='Purity versus source-class correspondence');axes[1].legend(fontsize=8)
    fig.suptitle('Target cluster routing diagnostic;3seed mean +/- SD; '+s['decision'])
    fig.savefig(OUT/'cluster_routing_transport_summary.png',dpi=180);plt.close(fig)
    print('\n'.join(lines[:35]),flush=True)

if __name__=='__main__':main()
