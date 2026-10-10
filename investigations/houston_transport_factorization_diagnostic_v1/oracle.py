"""Oracle 2x2 branches after predicted-factor freeze; no training/tuning."""
import csv
import json
import numpy as np
from common import HERE,OUT,ORACLE,DATA,SEEDS,GROUPS,sha,ahash,dump

def metric(y,p):
    cm=np.bincount(7*y+p,minlength=49).reshape(7,7);counts=cm.sum(1)
    recall=np.divide(np.diag(cm),counts,out=np.zeros(7,float),where=counts>0)
    return {'OA':float(100*np.trace(cm)/len(y)),'AA':float(100*recall[counts>0].mean()),
            'recall':(100*recall).tolist(),'confusion':cm.tolist()}

def changes(before,after,y,mask):
    b=before[mask]==y[mask];a=after[mask]==y[mask]
    recovered=int((~b&a).sum());damaged=int((b&~a).sum())
    return {'accuracy_before':float(b.mean()),'accuracy_after':float(a.mean()),'accuracy_delta':float((a.astype(float)-b).mean()),
        'recovered_n':recovered,'damaged_n':damaged,'recovery_fraction_of_subset':float(recovered/len(b)),
        'damage_fraction_of_subset':float(damaged/len(b)),
        'recovery_fraction_of_before_wrong':float(recovered/(~b).sum()) if (~b).any() else None,
        'damage_fraction_of_before_correct':float(damaged/b.sum()) if b.any() else None}

def masks(y,p):
    correct=y==p
    result={'all':np.ones(len(y),bool),'original_correct':correct,'original_error':~correct}
    for c in range(7):
        for name,m in [('all',np.ones(len(y),bool)),('original_correct',correct),('original_error',~correct)]:
            result[f'C{c+1}_{name}']=(y==c)&m
        for j in range(7):
            m=(y==c)&(p==j)
            if c!=j and m.any():result[f'C{c+1}_to_C{j+1}']=m
    return result

def effects(d):
    return {'routing_at_pseudo_stats_D01_minus_D00':d['D01']-d['D00'],
            'statistics_at_predicted_routing_D10_minus_D00':d['D10']-d['D00'],
            'routing_at_oracle_stats_D11_minus_D10':d['D11']-d['D10'],
            'statistics_at_oracle_routing_D11_minus_D01':d['D11']-d['D01'],
            'joint_D11_minus_D00':d['D11']-d['D00'],
            'interaction_D11_minus_D10_minus_D01_plus_D00':d['D11']-d['D10']-d['D01']+d['D00']}

def fixtures():
    d={'D00':10.,'D01':12.,'D10':13.,'D11':20.};r=effects(d)
    assert r['interaction_D11_minus_D10_minus_D01_plus_D00']==5
    assert r['joint_D11_minus_D00']==r['routing_at_pseudo_stats_D01_minus_D00']+r['statistics_at_predicted_routing_D10_minus_D00']+r['interaction_D11_minus_D10_minus_D01_plus_D00']
    y=np.array([0,1,2]);b=np.array([1,1,0]);a=np.array([0,2,2]);r=changes(b,a,y,np.ones(3,bool))
    assert r['recovered_n']==2 and r['damaged_n']==1

def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    fixtures()
    gate=json.loads((OUT/'ALL_PREDICTED_FACTORS_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened'] and not gate['oracle_G2_opened']
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    prov=json.loads((OUT/'provenance.json').read_text())
    for path,h in prov['code_sha256'].items():assert sha(path)==h
    print('Predicted statistics/routing/D00 hashes verified. Opening target GT for oracle branches.',flush=True)
    y=np.load(DATA/'target_labels.npy')
    assert ahash(y)==prov['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    old_gate=json.loads((ORACLE/'ORACLE_RESULTS_FROZEN.json').read_text())
    rows,metrics,factor_rows,subset_effects,audits=[],[],[],[],{}
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';z=np.load(folder/'target_rawZ.npy').astype(float)
        native=np.load(folder/'target_predictions.npy');head=np.load(folder/'frozen_head.npz')
        w=head['weight'].astype(float);b=head['bias'].astype(float)
        stats=np.load(folder/'predicted_factors.npz');source_means=stats['source_means']
        pseudo=stats['pseudo_target_means'];routing=stats['predicted_routing'];assert np.array_equal(routing,native)
        oracle_means=np.stack([z[y==c].mean(0) for c in range(7)])
        np.savez_compressed(folder/'oracle_factors.npz',oracle_target_class_means=oracle_means,oracle_routing=y)
        subsets=masks(y,native);preds={};seedmetric={}
        for group,means,route in [('D00',pseudo,routing),('D01',pseudo,y),('D10',oracle_means,routing),('D11',oracle_means,y)]:
            x=z+(source_means-means)[route];logits=x@w.T+b;pred=logits.argmax(1)
            if group=='D00':
                assert np.array_equal(x,np.load(folder/'D00_rawZ.npy'))
                existing=np.load(folder/'D00_readouts.npz')
                assert np.array_equal(logits,existing['logits']) and np.array_equal(pred,existing['predictions'])
            else:
                np.save(folder/f'{group}_rawZ.npy',x)
                np.savez_compressed(folder/f'{group}_readouts.npz',logits=logits,predictions=pred)
            if group=='D11':
                for name in ['G2_rawZ.npy','G2_oracle_readouts.npz']:
                    rel=f'seed_{seed}/{name}';assert sha(ORACLE/rel)==old_gate['files_sha256'][rel]
                refx=np.load(ORACLE/f'seed_{seed}/G2_rawZ.npy')
                ref=np.load(ORACLE/f'seed_{seed}/G2_oracle_readouts.npz')
                feature_error=float(abs(x-refx).max());logit_error=float(abs(logits-ref['frozen_classifier_logits']).max())
                assert feature_error<1e-12 and logit_error<1e-10
                assert np.array_equal(pred,ref['frozen_classifier_predictions'])
                audits[str(seed)]={'D00_F1_exact_replay':True,'D11_G2_features_max_error':feature_error,
                    'D11_G2_logits_max_error':logit_error,'D11_G2_predictions_exact':True}
            m=metric(y,pred);seedmetric[group]=m;preds[group]=pred
            metrics.append({'seed':seed,'group':group,'OA':m['OA'],'AA':m['AA']})
            for name,mask in subsets.items():
                if not mask.any():continue
                rows.append({'seed':seed,'group':group,'subset':name,'n':int(mask.sum()),**changes(native,pred,y,mask)})
            print(seed,group,'OA/AA',m['OA'],m['AA'],flush=True)
        for metric_name in ['OA','AA']:
            factor_rows.append({'seed':seed,'metric':metric_name,**effects({g:seedmetric[g][metric_name] for g in GROUPS})})
        for name,mask in subsets.items():
            if not mask.any():continue
            d={g:float(100*(preds[g][mask]==y[mask]).mean()) for g in GROUPS}
            subset_effects.append({'seed':seed,'subset':name,'n':int(mask.sum()),**effects(d)})
            for a,bgroup in [('D00','D01'),('D00','D10'),('D10','D11'),('D01','D11')]:
                before=preds[a][mask]==y[mask];after=preds[bgroup][mask]==y[mask]
                np.savez_compressed(folder/f'{a}_to_{bgroup}_all_sample_transitions.npz',
                    before_correct=preds[a]==y,after_correct=preds[bgroup]==y,
                    wrong_to_correct=(preds[a]!=y)&(preds[bgroup]==y),correct_to_wrong=(preds[a]==y)&(preds[bgroup]!=y)) if name=='all' else None
    csvwrite('per_seed_metrics.csv',metrics);csvwrite('subset_statistics.csv',rows)
    csvwrite('per_class.csv',[r for r in rows if r['subset'].startswith('C') and '_to_' not in r['subset']])
    csvwrite('confusion_pairs.csv',[r for r in rows if '_to_' in r['subset']])
    csvwrite('factor_effects_per_seed.csv',factor_rows);csvwrite('factor_effects_by_subset.csv',subset_effects)
    dump(OUT/'summary.json',{'metrics':metrics,'subsets':rows,'factor_effects':factor_rows,
        'subset_factor_effects':subset_effects,'replay_audits':audits,
        'oracle_GT_used_for_statistics_and_routing':True,'no_training_tuning_followup':True})
    report(metrics,rows,factor_rows,audits)
    for path,h in gate['files_sha256'].items():assert sha(OUT/path)==h
    for rec in prov['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_ORACLE_RESULTS_FROZEN.json',{'files_sha256':artifacts,'predicted_factors_frozen_before_GT':True,
        'native_inputs_predicted_factors_D00_checkpoints_unchanged':True,'D11_G2_prediction_replay_all_seeds':True,
        'target_GT_used_only_for_oracle_construction_and_scoring':True,'no_training_tuning_followup':True,
        'target_labels_sha256':sha(DATA/'target_labels.npy'),'fixtures_passed':True})

def report(metrics,rows,factors,audits):
    def vals(group,key):return np.array([r[key] for r in metrics if r['group']==group],float)
    def sv(group,subset,key):return np.array([r[key] for r in rows if r['group']==group and r['subset']==subset and r[key] is not None],float)
    def fmt(a,pct=False):
        a=np.asarray(a)*(100 if pct else 1)
        return f'{a.mean():.5f} +/- {a.std(ddof=1):.5f}' if len(a)>1 else (str(a[0]) if len(a) else 'NA')
    lines=['Houston transport factorization V1: fixed hard F1 statistics/routing, original C3seeds, epoch200 raw CLS.',
        'Predicted statistics/routing/native inputs/D00 frozen before target GT. No training/backward/tuning/follow-up.',
        'D00 pseudo/predicted; D01 pseudo/oracle-routing; D10 oracle-statistics/predicted; D11 oracle/oracle.',
        'D00 exact feature/logit/prediction replay ofF1; D11 G2 prediction replay exact (float64 arithmetic tolerances documented).',
        'All means +/- sample SD across3 optimization seeds; percentage changes use same-seed subtraction.','']
    for group in GROUPS:
        lines.append(f'{group}: OA={fmt(vals(group,"OA"))}%; AA={fmt(vals(group,"AA"))}%; '+
            'original-error recovery='+fmt(sv(group,'original_error','recovery_fraction_of_subset'),True)+'%; '+
            'original-correct damage='+fmt(sv(group,'original_correct','damage_fraction_of_subset'),True)+'%')
    aggregate=[]
    lines.extend(['','Factor effects (percentage points):'])
    for metric_name in ['OA','AA']:
        rr=[r for r in factors if r['metric']==metric_name]
        for key in rr[0]:
            if key in ['seed','metric']:continue
            values=[r[key] for r in rr]
            lines.append(metric_name+' '+key+': '+fmt(values)+' pp; seeds='+str(values))
            aggregate.append({'metric':metric_name,'effect':key,'mean':float(np.mean(values)),
                              'sample_std':float(np.std(values,ddof=1)),'values':values})
    csvwrite('factor_effects_seed_aggregate.csv',aggregate)
    for subset in ['C1_all','C7_all','C7_to_C6','C1_original_error','C7_original_error']:
        lines.extend(['',subset])
        for group in GROUPS:
            lines.append(group+' accuracy='+fmt(sv(group,subset,'accuracy_after'),True)+'%; recovered n='+str(sv(group,subset,'recovered_n').astype(int).tolist())+'; damaged n='+str(sv(group,subset,'damaged_n').astype(int).tolist()))
    limits=['True-class routing and same-evaluation-set true means are oracle answers, not deployable UDA.',
        'Statistics/routing effects depend on the other factor; interaction prevents an additive causal attribution.',
        'Single-factor weak gains do not show the factor is unnecessary when the other factor is inaccurate.',
        'This fixed hard one-step translation cannot establish dominance for soft/other estimators or true physical shifts.',
        'Same-target-set means are optimistic; one scene pair,3optimization seeds; no significance/generalization claim.']
    lines.extend(['','Replay audits:',json.dumps(audits),'','Limitations:',*limits])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    findings=['Houston transport factorization V1：hard统计与hard routing的固定2×2 oracle解耦。',
        'D00重放F1，D11重放G2。predicted factors/D00在读取GT前全部冻结。无训练、反传、调参或后续实验。','']
    findings.extend(lines[6:10])
    findings.extend(['','同seed factor effects，mean±sampleSD：'])
    findings.extend(r for r in lines if r.startswith('OA ') or r.startswith('AA '))
    findings.extend(['','重点类别准确率：'])
    for subset in ['C1_all','C7_all','C7_to_C6']:
        findings.append(subset+': '+'; '.join(g+'='+fmt(sv(g,subset,'accuracy_after'),True)+'%' for g in GROUPS))
    findings.extend(['','解释边界：',*limits])
    (OUT/'FINDINGS.txt').write_text('\n'.join(findings)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    grid=[['D00','D01'],['D10','D11']]
    for ax,key in zip(axes,['OA','AA']):
        mean=np.array([[vals(g,key).mean() for g in row] for row in grid]);sd=np.array([[vals(g,key).std(ddof=1) for g in row] for row in grid])
        im=ax.imshow(mean,vmin=0,vmax=100,cmap='YlGnBu')
        ax.set(title=f'Frozen classifier {key}',xlabel='Sample routing',ylabel='Target class statistics',xticks=[0,1],yticks=[0,1],xticklabels=['Predicted','Oracle GT'],yticklabels=['Pseudo','Oracle GT'])
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{grid[i][j]}\n{mean[i,j]:.2f} +/- {sd[i,j]:.2f}%',ha='center',va='center',color='white' if mean[i,j]>70 else 'black')
    fig.colorbar(im,ax=axes,label='Accuracy (%)');fig.suptitle('Houston hard mean transport factorization:3seed mean +/- SD; oracle diagnosis only')
    fig.savefig(OUT/'factorization_2x2.png',dpi=180);plt.close(fig)
    print('\n'.join(findings[:22]),flush=True)

if __name__=='__main__':main()
