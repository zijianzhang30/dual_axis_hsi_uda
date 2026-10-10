"""GT scoring of frozen routers, availability ceilings, and majority transport."""
import csv
import json
import numpy as np
from common import HERE,OUT,DATA,SEEDS,KS,sha,ahash,dump

def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def metric(y,p):
    cm=np.bincount(7*y+p,minlength=49).reshape(7,7);n=cm.sum(1)
    r=np.divide(np.diag(cm),n,out=np.zeros(7,float),where=n>0)
    return {'OA':float(100*np.trace(cm)/len(y)),'AA':float(100*r[n>0].mean()),
            'represented_classes':int((n>0).sum()),'class_recall':(100*r).tolist(),'confusion':cm.tolist()}

def selector_metrics(y,correct):
    return {'OA':float(100*correct.mean()),'AA':float(np.mean([100*correct[y==c].mean() for c in range(7) if (y==c).any()]))}

def masks(y,p):
    right=y==p
    result={'all':np.ones(len(y),bool),'R0_correct':right,'R0_error':~right}
    for c in range(7):
        for name,m in [('all',np.ones(len(y),bool)),('R0_correct',right),('R0_error',~right)]:
            result[f'C{c+1}_{name}']=(y==c)&m
        for j in range(7):
            m=(y==c)&(p==j)
            if c!=j and m.any():result[f'C{c+1}_to_C{j+1}']=m
    return result

def transitions(before,after,y,mask):
    b=before[mask]==y[mask];a=after[mask]==y[mask]
    gain=int((~b&a).sum());loss=int((b&~a).sum())
    return {'accuracy_before':float(b.mean()),'accuracy_after':float(a.mean()),'accuracy_delta':float(a.mean()-b.mean()),
        'recovered_n':gain,'damaged_n':loss,'recovery_fraction_of_subset':float(gain/len(b)),
        'damage_fraction_of_subset':float(loss/len(b)),
        'recovery_fraction_of_before_wrong':float(gain/(~b).sum()) if (~b).any() else None,
        'damage_fraction_of_before_correct':float(loss/b.sum()) if b.any() else None}

def fixtures():
    y=np.array([0,1,2]);r0=np.array([1,1,0]);a=np.array([0,1,0]);b=np.array([1,1,2])
    correct=np.stack([r0==y,a==y,b==y],axis=1)
    assert correct.any(1).all() and (correct.any(1)[r0!=y]).mean()==1
    assert ((a==y)&(b!=y)&(r0!=y)).sum()==1
    assert ((b==y)&(a!=y)&(r0!=y)).sum()==1
    r=transitions(r0,np.array([0,2,2]),y,np.ones(3,bool))
    assert r['recovered_n']==2 and r['damaged_n']==1

def main():
    fixtures()
    gate=json.loads((OUT/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_constructor']
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    print('All router/majority/transport hashes verified. Opening target GT offline.',flush=True)
    y=np.load(DATA/'target_labels.npy')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    routing_rows,routing_subsets,union_rows,pair_rows,unique_rows,patterns,transport_rows,transport_subsets,tie_rows=[],[],[],[],[],[],[],[],[]
    details={}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';r0=np.load(folder/'R0_predictions.npy');subset=masks(y,r0);errors=r0!=y
        d00=np.load(folder/'D00_readouts.npz')['predictions'];details[str(seed)]={}
        for k in KS:
            routers={r:np.load(folder/(f'{r}_k{k}_predictions.npy' if r in ['R2','R4','R6'] else f'{r}_predictions.npy')) for r in ['R0','R1','R2','R3','R4','R5','R6']}
            maj=np.load(folder/f'Majority_k{k}_routing.npz');routers['Majority']=maj['predictions']
            core=np.stack([routers[f'R{i}']==y for i in range(5)],axis=1)
            auxiliary=np.stack([routers[f'R{i}']==y for i in range(7)],axis=1)
            assert ((routers['Majority']==y)&~core.any(1)).sum()==0
            for name,pred in routers.items():
                m=metric(y,pred);details[str(seed)][f'{name}_k{k}']=m
                routing_rows.append({'seed':seed,'k':k,'router':name,'OA':m['OA'],'AA':m['AA'],
                    'R0_error_correct_percent':float(100*(pred[errors]==y[errors]).mean()),
                    'R0_correct_damaged_percent':float(100*(pred[~errors]!=y[~errors]).mean())})
                for sub,mask in subset.items():
                    if mask.any():routing_subsets.append({'seed':seed,'k':k,'router':name,'subset':sub,'n':int(mask.sum()),
                                                         'correct_fraction':float((pred[mask]==y[mask]).mean())})
            for name,c in [('primary_R0_R4',core),('auxiliary_R0_R6',auxiliary)]:
                available=c.any(1);m=selector_metrics(y,available)
                union_rows.append({'seed':seed,'k':k,'router_set':name,'OA_ceiling':m['OA'],'AA_ceiling':m['AA'],
                    'R0_error_available_n':int((errors&available).sum()),'R0_error_available_percent':float(100*available[errors].mean()),
                    'R0_error_unavailable_n':int((errors&~available).sum())})
                for sub,mask in subset.items():
                    if mask.any():routing_subsets.append({'seed':seed,'k':k,'router':'OracleSelector_'+name,'subset':sub,'n':int(mask.sum()),
                                                         'correct_fraction':float(available[mask].mean())})
                np.savez_compressed(folder/f'{name}_k{k}_GT_availability.npz',correct_router_flags=c,target_labels=y,
                                    any_router_correct=available,R0_error=errors)
            for a in range(5):
                ca=core[:,a]
                unique=errors&ca&~np.delete(core,a,axis=1).any(1)
                unique_rows.append({'seed':seed,'k':k,'router':f'R{a}','unique_on_R0_error_n':int(unique.sum()),
                    'unique_on_R0_error_percent':float(100*unique.sum()/errors.sum()),
                    'correct_on_R0_error_percent':float(100*ca[errors].mean())})
                for b in range(5):
                    cb=core[:,b];gain=errors&ca&~cb;both=errors&ca&cb;either=errors&(ca|cb)
                    pair_rows.append({'seed':seed,'k':k,'router_A':f'R{a}','router_B':f'R{b}','R0_error_n':int(errors.sum()),
                        'A_extra_over_B_n':int(gain.sum()),'A_extra_over_B_percent':float(100*gain.sum()/errors.sum()),
                        'A_correct_given_B_wrong_percent':float(100*gain.sum()/(errors&~cb).sum()) if (errors&~cb).any() else None,
                        'both_correct_percent':float(100*both.sum()/errors.sum()),'either_correct_percent':float(100*either.sum()/errors.sum())})
            bitmask=(core.astype(np.int64)*(1<<np.arange(5))[None,:]).sum(1)
            for code in range(32):
                count=int((errors&(bitmask==code)).sum())
                if count:patterns.append({'seed':seed,'k':k,'bitmask':code,'R0_to_R4_correct_bits':''.join(str((code>>j)&1) for j in range(5)),
                                          'R0_error_n':count,'fraction_of_R0_errors':float(count/errors.sum())})
            tp=np.load(folder/f'Majority_k{k}_transport_readouts.npz')['predictions']
            for name,pred in [('D00',d00),('Majority_transport',tp)]:
                m=metric(y,pred);before=metric(y,d00)
                transport_rows.append({'seed':seed,'k':k,'transport':name,'OA':m['OA'],'AA':m['AA'],
                                      'delta_OA_vs_D00':m['OA']-before['OA'],'delta_AA_vs_D00':m['AA']-before['AA']})
                for sub,mask in subset.items():
                    if mask.any():transport_subsets.append({'seed':seed,'k':k,'transport':name,'subset':sub,'n':int(mask.sum()),
                        **transitions(r0,pred,y,mask),
                        'delta_accuracy_vs_D00':float((pred[mask]==y[mask]).mean()-(d00[mask]==y[mask]).mean())})
            for sub,mask in subset.items():
                if mask.any():tie_rows.append({'seed':seed,'k':k,'subset':sub,'n':int(mask.sum()),
                                              'majority_tie_fraction':float(maj['tied'][mask].mean())})
            print(seed,'k',k,'primary error coverage',union_rows[-2]['R0_error_available_percent'],
                  'majority routing error correct',routing_rows[-1]['R0_error_correct_percent'],
                  'majority transport OA/AA',transport_rows[-1]['OA'],transport_rows[-1]['AA'],flush=True)
    csvwrite('routing_per_seed.csv',routing_rows);csvwrite('routing_subset_statistics.csv',routing_subsets)
    csvwrite('routing_per_class.csv',[r for r in routing_subsets if r['subset'].startswith('C') and '_to_' not in r['subset']])
    csvwrite('routing_confusion_pairs.csv',[r for r in routing_subsets if '_to_' in r['subset']])
    csvwrite('oracle_selector_coverage.csv',union_rows);csvwrite('pairwise_complementarity.csv',pair_rows)
    csvwrite('unique_router_coverage.csv',unique_rows);csvwrite('R0_error_correctness_patterns.csv',patterns)
    csvwrite('transport_per_seed.csv',transport_rows);csvwrite('transport_subset_statistics.csv',transport_subsets)
    csvwrite('transport_per_class.csv',[r for r in transport_subsets if r['subset'].startswith('C') and '_to_' not in r['subset']])
    csvwrite('transport_confusion_pairs.csv',[r for r in transport_subsets if '_to_' in r['subset']])
    csvwrite('majority_tie_audit.csv',tie_rows)
    limits=['Oracle selector is GT-assisted signal availability, not a learned/no-label router or transformed-head ceiling.',
        'PrimaryR0-R4 and auxiliaryV-inclusive R0-R6 reported separately; V never enters fixed majority.',
        'k5/15 and voting rule fixed before GT; all outcomes retained, no best-router/rule choice.',
        'Centroid and kNN votes share source features/labels and can be highly correlated.',
        'D00 pseudo target means are unchanged; only majority sample routing changes.',
        'Native R0 correctness defines subsets; transport correctness and router correctness are distinct.',
        'One scene pair,3 optimization seeds, overlapping patches; no causal/significance/generalization claim.']
    summary={'routing_metrics':routing_rows,'routing_subsets':routing_subsets,'oracle_selector':union_rows,
        'pairwise_complementarity':pair_rows,'unique_router_coverage':unique_rows,'correctness_patterns':patterns,
        'transport_metrics':transport_rows,'transport_subsets':transport_subsets,'tie_audit':tie_rows,'details':details,'limitations':limits}
    dump(OUT/'summary.json',summary)
    report(summary)
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'all_frozen_features_routers_transport_checkpoints_unchanged':True,
        'target_GT_only_after_all_routing_transport_freeze':True,'no_training_tuning_followup':True,
        'summary_sha256':sha(OUT/'summary.json'),'report_sha256':sha(OUT/'REPORT.txt'),
        'findings_sha256':sha(OUT/'FINDINGS.txt'),'figure_sha256':sha(OUT/'routing_signal_summary.png')})

def report(summary):
    def select(rows,**filters):return [r for r in rows if all(r[key]==val for key,val in filters.items())]
    def vals(rows,key,**filters):return np.array([r[key] for r in select(rows,**filters) if r[key] is not None],float)
    def fmt(a,pct=False):
        a=np.asarray(a)*(100 if pct else 1)
        return f'{a.mean():.5f} +/- {a.std(ddof=1):.5f}' if len(a)>1 else (str(a[0]) if len(a) else 'NA')
    rr=summary['routing_metrics'];rs=summary['routing_subsets'];uu=summary['oracle_selector']
    tm=summary['transport_metrics'];ts=summary['transport_subsets']
    lines=['Houston routing signal V1: frozen original C3seeds epoch200, no training/tuning/follow-up.',
        'All routers/majority/features/predictions frozen before target GT; source2393 train, target52901.',
        'R0 Z classifier; R1Z centroid/R2Z kNN; R3T centroid/R4T kNN; R5V centroid/R6V kNN auxiliary.',
        'Majority votesR1-R4 only, unique plurality wins, highest-count ties fallbackR0; k5/15 separately fixed.',
        'Mean +/- sampleSD of3seed metrics; selector availability is oracle, not a deployable policy.','']
    for k in KS:
        lines.append(f'k={k} routing metrics:')
        for r in ['R0','R1','R2','R3','R4','R5','R6','Majority']:
            lines.append(r+' OA='+fmt(vals(rr,'OA',k=k,router=r))+'%; AA='+fmt(vals(rr,'AA',k=k,router=r))+
                '%; correct onR0 errors='+fmt(vals(rr,'R0_error_correct_percent',k=k,router=r))+
                '%; damage onR0 correct='+fmt(vals(rr,'R0_correct_damaged_percent',k=k,router=r))+'%')
        for s in ['primary_R0_R4','auxiliary_R0_R6']:
            lines.append(s+' selector OA='+fmt(vals(uu,'OA_ceiling',k=k,router_set=s))+'%; AA='+fmt(vals(uu,'AA_ceiling',k=k,router_set=s))+
                '%; R0-error coverage='+fmt(vals(uu,'R0_error_available_percent',k=k,router_set=s))+'%')
        for t in ['D00','Majority_transport']:
            lines.append(t+' OA='+fmt(vals(tm,'OA',k=k,transport=t))+'%; AA='+fmt(vals(tm,'AA',k=k,transport=t))+
                '%; same-seed OA delta='+fmt(vals(tm,'delta_OA_vs_D00',k=k,transport=t))+'pp; AA delta='+fmt(vals(tm,'delta_AA_vs_D00',k=k,transport=t))+'pp')
            lines.append('  original error recovery='+fmt(vals(ts,'recovery_fraction_of_subset',k=k,transport=t,subset='R0_error'),True)+
                '%; original correct damage='+fmt(vals(ts,'damage_fraction_of_subset',k=k,transport=t,subset='R0_correct'),True)+'%')
        lines.append('')
    for subset in ['C1_all','C7_all','C1_R0_error','C7_R0_error','C7_to_C6']:
        lines.append(subset)
        for k in KS:
            lines.append(f'k{k}: '+'; '.join(r+'='+fmt(vals(rs,'correct_fraction',k=k,router=r,subset=subset),True)+'%' for r in ['R0','R1','R2','R3','R4','Majority','OracleSelector_primary_R0_R4']))
    lines.extend(['','Limitations:',*summary['limitations']]);(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    findings=['Houston routing signal V1：冻结source-anchored routing诊断。',
        '全部router/多数票/固定D00统计下的transport在读取target GT之前冻结。无训练、调参或后续实验。',
        'R0–R4是主routers；V的R5/R6为固定辅助，不能参与多数票。oracle-selector仅表示正确信号可用性。','']
    findings.extend(lines[6:])
    (OUT/'FINDINGS.txt').write_text('\n'.join(findings)+'\n')
    # Mean/sampleSD table preserves the explicit endpoint scale (percent vs fraction).
    agg=[]
    for k in KS:
        for router in ['R0','R1','R2','R3','R4','R5','R6','Majority']:
            for key in ['OA','AA','R0_error_correct_percent','R0_correct_damaged_percent']:
                a=vals(rr,key,k=k,router=router)
                agg.append({'k':k,'router':router,'metric':key,'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)),'values':a.tolist()})
    csvwrite('routing_seed_aggregate.csv',agg)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    labels=['R1','R2','R3','R4','Majority','Oracle union']
    for offset,k,color in [(-.18,5,'#457b9d'),(.18,15,'#e9c46a')]:
        arrays=[vals(rr,'R0_error_correct_percent',k=k,router=r) for r in labels[:-1]]+[vals(uu,'R0_error_available_percent',k=k,router_set='primary_R0_R4')]
        a=np.array(arrays);axes[0].bar(np.arange(6)+offset,a.mean(1),.36,yerr=a.std(1,ddof=1),capsize=3,label=f'k{k}',color=color)
    axes[0].set(title='Correct routing on R0 errors',ylabel='Original errors (%)',xticks=range(6),xticklabels=labels);axes[0].legend()
    for offset,k,color in [(-.18,5,'#457b9d'),(.18,15,'#e9c46a')]:
        a=np.array([vals(tm,'OA',k=k,transport=t) for t in ['D00','Majority_transport']])
        axes[1].bar(np.arange(2)+offset,a.mean(1),.36,yerr=a.std(1,ddof=1),capsize=3,label=f'k{k}',color=color)
    axes[1].set(title='Fixed-statistics transport OA',ylabel='Accuracy (%)',xticks=[0,1],xticklabels=['D00/R0 route','Majority route'],ylim=(0,100));axes[1].legend()
    for offset,key,sub,color,label in [(-.18,'recovery_fraction_of_subset','R0_error','#2a9d8f','Error recovery'),(.18,'damage_fraction_of_subset','R0_correct','#e76f51','Correct damage')]:
        a=np.array([vals(ts,key,k=k,transport='Majority_transport',subset=sub) for k in KS])*100
        axes[2].bar(np.arange(2)+offset,a.mean(1),.36,yerr=a.std(1,ddof=1),capsize=3,label=label,color=color)
    axes[2].set(title='Majority transport recovery/damage',ylabel='Subset samples (%)',xticks=[0,1],xticklabels=['k5','k15']);axes[2].legend()
    fig.suptitle('Houston frozen routing signals,3seed mean +/- SD; fixed majority; oracle union is availability only')
    fig.savefig(OUT/'routing_signal_summary.png',dpi=180);plt.close(fig)
    print('\n'.join(findings[:30]),flush=True)

if __name__=='__main__':main()
