import csv, hashlib, importlib.util, json, shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_boundary_geometric_stress_diagnostic_v1'
BASE=ROOT/'results/houston_transport_factorization_diagnostic_v1'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS=(2100,2101,2102)
PAIRS=[(a,b) for a in range(7) for b in range(a+1,7)]
spec=importlib.util.spec_from_file_location('boundary_utils',ROOT/'investigations/houston_target_boundary_signal_diagnostic_v1/run.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
sha,dump,aggregate=u.sha,u.dump,u.aggregate

def write(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def stats(x):
    if not len(x):return {'n':0,**{k:None for k in ['min','p05','p25','median','p75','p95','max','mean','std']}}
    q=np.quantile(x,[0,.05,.25,.5,.75,.95,1])
    return dict(n=len(x),**dict(zip(['min','p05','p25','median','p75','p95','max'],map(float,q))),mean=float(x.mean()),std=float(x.std(ddof=1)) if len(x)>1 else None)

def main():
    OUT.mkdir(exist_ok=False)
    gate=json.loads((BASE/'ALL_PREDICTED_FACTORS_FROZEN.json').read_text())
    old=json.loads((BASE/'provenance.json').read_text())
    checks=old['historical_metadata']['checkpoints']
    for rec in checks.values():assert sha(rec['path'])==rec['sha256']
    def copy(rel):
        assert sha(BASE/rel)==gate['files_sha256'][rel]
        dest=OUT/rel;dest.parent.mkdir(exist_ok=True);shutil.copyfile(BASE/rel,dest)
        assert sha(dest)==gate['files_sha256'][rel]
    copy('source_labels.npy');copy('source_train_mask.npy')
    sy=np.load(OUT/'source_labels.npy');rows=[];distributions=[];audit=[]
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}'
        for name in ['source_rawZ.npy','target_rawZ.npy','frozen_head.npz','source_logits.npy','target_logits.npy','target_predictions.npy','source_coords.npy','target_coords.npy','source_sample_indices.npy','target_sample_indices.npy']:
            copy(f'seed_{seed}/{name}')
        head=np.load(folder/'frozen_head.npz');w=head['weight'].astype(float);b=head['bias'].astype(float)
        normals=np.array([w[a]-w[bb] for a,bb in PAIRS]);norms=np.linalg.norm(normals,axis=1);assert np.all(norms>0)
        bias=np.array([b[a]-b[bb] for a,bb in PAIRS]);distance={}
        for domain in ['source','target']:
            z=np.load(folder/f'{domain}_rawZ.npy').astype(float);native=np.load(folder/f'{domain}_logits.npy')
            replay=z@w.T+b;err=float(np.max(np.abs(native-replay)));assert err<2e-5
            assert np.array_equal(np.argmax(native,axis=1),np.argmax(replay,axis=1))
            distance[domain]=np.abs(z@normals.T+bias)/norms
            np.save(folder/f'{domain}_pair_distances.npy',distance[domain])
            audit.append({'seed':seed,'domain':domain,'n':len(z),'replay_max_logit_error':err})
        logits=np.load(folder/'target_logits.npy');order=np.argsort(-logits,axis=1,kind='stable');p,q=order[:,0],order[:,1]
        assert np.array_equal(p,np.load(folder/'target_predictions.npy'))
        low,high=np.minimum(p,q),np.maximum(p,q);counts=[];ratios=[];stress=[]
        for j,(a,bb) in enumerate(PAIRS):
            tm=(low==a)&(high==bb);sm=(sy==a)|(sy==bb)
            ts=stats(distance['target'][tm,j]);ss=stats(distance['source'][sm,j])
            ratio=ts['median']/ss['median'] if ts['n'] and ss['median']>0 else None
            st=ss['median']/ts['median'] if ts['n'] and ts['median']>0 else None
            counts.append(ts['n']);ratios.append(ratio);stress.append(st)
            rows.append({'seed':seed,'pair':f'C{a+1}/C{bb+1}','normal_norm':float(norms[j]),'competition_n':ts['n'],'competition_percent':100*ts['n']/len(p),
                'target_median':ts['median'],'source_pooled_median':ss['median'],'target_source_median_ratio':ratio,'stress_source_target_median_ratio':st})
            for population,x in [('target_top2_pair',distance['target'][tm,j]),('source_true_pair_pooled',distance['source'][sm,j]),
                (f'source_true_C{a+1}',distance['source'][sy==a,j]),(f'source_true_C{bb+1}',distance['source'][sy==bb,j])]:
                distributions.append({'seed':seed,'pair':f'C{a+1}/C{bb+1}','population':population,**stats(x)})
        counts=np.array(counts);ratios=np.array([np.nan if x is None else x for x in ratios]);stress=np.array([np.nan if x is None else x for x in stress])
        ranks={'frequency':np.lexsort((np.arange(21),-counts)),
            'stress':np.lexsort((np.arange(21),np.where(np.isfinite(stress),-stress,np.inf))),
            'ratio':np.lexsort((np.arange(21),np.where(np.isfinite(ratios),ratios,np.inf)))}
        assert counts.sum()==len(p)
        for j in range(21):
            for method,rank in ranks.items():rows[-21+j][method+'_rank']=int(np.flatnonzero(rank==j)[0]+1)
        np.savez_compressed(folder/'unlabeled_boundary_signals.npz',top1=p,top2=q,pair_low=low,pair_high=high,counts=counts,ratio=ratios,stress=stress,**{k+'_ranking':v for k,v in ranks.items()})
    write('unlabeled_pair_statistics_per_seed.csv',rows);write('distance_distributions_per_seed.csv',distributions);write('classifier_replay_audit.csv',audit)
    dump(OUT/'provenance.json',{'historical_metadata':old['historical_metadata'],'base_manifest_sha256':sha(BASE/'ALL_PREDICTED_FACTORS_FROZEN.json'),
        'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PROTOCOL.txt'),'utility_sha256':sha(Path(u.__file__))})
    frozen={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_BOUNDARY_STRESS_FROZEN.json',{'target_GT_opened':False,'all21_pairs_all3_seeds_complete':True,'no_training_or_tuning':True,'files_sha256':frozen})
    for rel,h in frozen.items():assert sha(OUT/rel)==h
    print('All inputs, geometric distances, stress and frequency rankings frozen; now target GT scoring.',flush=True)
    y=np.load(DATA/'target_labels.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()==old['historical_metadata']['domains']['target']['labels_sha256']
    results=[];pair_results=[];classes=[]
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';coords=np.load(f/'target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
        s=np.load(f/'unlabeled_boundary_signals.npz');p=s['top1'];errors=p!=y;lo,hi=np.minimum(p,y),np.maximum(p,y)
        ec=np.array([(errors&(lo==a)&(hi==bb)).sum() for a,bb in PAIRS]);assert ec.sum()==errors.sum()
        for method,values in [('frequency',s['counts']),('stress',s['stress']),('ratio',s['ratio'])]:
            valid=np.isfinite(values);rank=s[method+'_ranking'];sel=rank[:3]
            covered=sum(ec[j] for j in sel)
            local=sum(((errors)&(y==s['top2'])&(s['pair_low']==PAIRS[j][0])&(s['pair_high']==PAIRS[j][1])).sum() for j in sel)
            results.append({'seed':seed,'method':method,'valid_pairs':int(valid.sum()),'Spearman_vs_true_error_count':float(spearmanr(values[valid],ec[valid]).statistic),
                'top3_pairs':';'.join(f'C{PAIRS[j][0]+1}/C{PAIRS[j][1]+1}' for j in sel),'top3_true_error_edge_coverage_percent':100*covered/errors.sum(),
                'top3_same_sample_correct_top2_coverage_percent':100*local/errors.sum(),'original_errors':int(errors.sum())})
        for j,(a,bb) in enumerate(PAIRS):
            tm=(s['pair_low']==a)&(s['pair_high']==bb)
            original=next(r for r in rows if r['seed']==seed and r['pair']==f'C{a+1}/C{bb+1}')
            pair_results.append({**original,'true_error_pair_n':int(ec[j]),'true_error_share_percent':100*ec[j]/errors.sum(),
                'competition_sample_error_percent':100*errors[tm].mean() if tm.any() else None})
        for c in range(7):classes.append({'seed':seed,'class':f'C{c+1}','n':int((y==c).sum()),'original_error_n':int((errors&(y==c)).sum())})
    write('scoring_per_seed.csv',results);write('pair_scoring_per_seed.csv',pair_results);write('class_errors_per_seed.csv',classes)
    summary=aggregate(results,['method'],['Spearman_vs_true_error_count','top3_true_error_edge_coverage_percent','top3_same_sample_correct_top2_coverage_percent'])
    pairs=aggregate(pair_results,['pair'],['competition_percent','target_median','source_pooled_median','target_source_median_ratio','stress_source_target_median_ratio','frequency_rank','stress_rank','true_error_pair_n','true_error_share_percent','competition_sample_error_percent'])
    write('scoring_seed_aggregate.csv',summary);write('pair_scoring_seed_aggregate.csv',pairs)
    write('distance_distributions_seed_aggregate.csv',aggregate(distributions,['pair','population'],['n','min','p05','p25','median','p75','p95','max','mean','std']))
    dump(OUT/'summary.json',{'metrics':summary,'pairs':pairs,'per_seed':results,'no_training_or_automatic_followup':True})
    report(summary,pairs,results)
    for rel,h in frozen.items():assert sha(OUT/rel)==h
    for rec in checks.values():assert sha(rec['path'])==rec['sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'frozen_files_and_checkpoints_unchanged':True,'target_GT_read_only_after_freeze':True,'no_training_tuning_followup':True})
    dump(OUT/'FINAL_DELIVERY.json',{'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FINAL_DELIVERY.json'}})

def report(summary,pairs,results):
    lines=['Houston boundary geometric stress diagnostic; originalC3seeds epoch200; frozen/offline only.',
        'Native rawZ feature-space distances; no L2 normalization. Source reference all2520 labeled samples; pair sides also separate.',
        'stress=source pooled median/target top2-pair median, descending; reciprocal ratio ascending. Frequency separate; no combined score.',
        'All distributions/rankings hash-frozen before target GT; mean+/-sampleSD over3modelseeds.','']
    for r in summary:lines.append(str(r))
    lines.append('\nPer-seed Top3:')
    for r in results:lines.append(str(r))
    lines.append('\nPrespecified focus pairs:')
    for r in pairs:
        if r['pair'] in ['C5/C6','C6/C7','C4/C6']:lines.append(str(r))
    freq=next(r for r in summary if r['method']=='frequency');stress=next(r for r in summary if r['method']=='stress')
    lines.extend(['\nInterpretation:',f"Geometric stress vs frequency: Spearman {stress['Spearman_vs_true_error_count_mean']:.4f} vs {freq['Spearman_vs_true_error_count_mean']:.4f}; Top3 true-error-edge coverage {stress['top3_true_error_edge_coverage_percent_mean']:.2f}% vs {freq['top3_true_error_edge_coverage_percent_mean']:.2f}%.",
        'This measures pair-hyperplane proximity relative to labeled source, not target correctness or the required boundary-movement direction.',
        'Infinite pair hyperplanes may not be active multiclass boundaries. Absolute distances erase side/sign; no claim of safety or training benefit.',
        'Ratios can reflect source pair asymmetry, class composition and source/target feature geometry. Source side distributions and all21pairs are retained.',
        'No formula/pair/threshold chosen using GT; no training, backpropagation, parameter search or automatic follow-up.'])
    text='\n'.join(lines)+'\n';(OUT/'REPORT.txt').write_text(text);(OUT/'FINDINGS.txt').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for seed in SEEDS:
        rs=[r for r in results if r['seed']==seed]
        ax[1].plot([0,1],[next(r['top3_true_error_edge_coverage_percent'] for r in rs if r['method']==m) for m in ['frequency','stress']],marker='o',label=str(seed))
    for r in pairs:
        ax[0].scatter(r['competition_percent_mean'],r['stress_source_target_median_ratio_mean'],s=20+4*r['true_error_share_percent_mean'],alpha=.7)
        ax[0].annotate(r['pair'],(r['competition_percent_mean'],r['stress_source_target_median_ratio_mean']),fontsize=7)
    ax[0].set(xlabel='Target top2-pair frequency (%)',ylabel='Geometric stress: source median / target median',title='All21 pairs; bubble size = post-freeze GT error share')
    ax[1].set(xticks=[0,1],xticklabels=['Frequency Top3','Geometric-stress Top3'],ylabel='Aggregate true-error-edge coverage (%)',ylim=(0,100));ax[1].legend()
    fig.savefig(OUT/'geometric_stress_vs_frequency.png',dpi=180);plt.close(fig)
    print(text,flush=True)

if __name__=='__main__':main()
