"""Small frozen top1/top2 aggregate-signal diagnostic, never trains/selects boundaries."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_target_boundary_signal_diagnostic_v1'
BASE=ROOT/'results/houston_routing_signal_diagnostic_v1'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
SEEDS=(2100,2101,2102)
PAIRS=[(a,b) for a in range(7) for b in range(a+1,7)]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvwrite(name,rows):
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def aggregate(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for group,rs in groups.items():
        x=dict(zip(keys,group))
        for key in metrics:
            a=np.array([r[key] for r in rs if r.get(key) is not None],float)
            x[key+'_mean']=float(a.mean()) if len(a) else None
            x[key+'_std']=float(a.std(ddof=1)) if len(a)>1 else None
        result.append(x)
    return result

def main():
    OUT.mkdir(exist_ok=False)
    gate=json.loads((BASE/'ALL_ROUTING_TRANSPORT_FROZEN.json').read_text());assert not gate['target_GT_opened_by_constructor']
    old=json.loads((BASE/'provenance.json').read_text());allpressure=[]
    prov={'checkpoints':old['seeds'],'historical_metadata':old['historical_metadata'],
          'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PROTOCOL.txt')}
    for seed in SEEDS:
        f=OUT/f'seed_{seed}';f.mkdir();rec=old['seeds'][str(seed)]
        assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
        for name in ['target_logits.npy','target_predictions.npy','target_coords.npy','target_sample_indices.npy','source_logits.npy']:
            rel=f'seed_{seed}/{name}';assert sha(BASE/rel)==gate['files_sha256'][rel]
            shutil.copyfile(BASE/rel,f/name);assert sha(f/name)==gate['files_sha256'][rel]
        for domain in ['source','target']:
            logits=np.load(f/f'{domain}_logits.npy').astype(float)
            # Stable ties favor smallest class ID; preserve native top1 exactly.
            order=np.argsort(-logits,axis=1,kind='stable');top1=order[:,0];top2=order[:,1]
            margin=logits[np.arange(len(logits)),top1]-logits[np.arange(len(logits)),top2]
            low=np.minimum(top1,top2);high=np.maximum(top1,top2)
            counts=np.array([((low==a)&(high==b)).sum() for a,b in PAIRS])
            ranking=np.lexsort((np.arange(21),-counts))
            if domain=='target':assert np.array_equal(top1,np.load(f/'target_predictions.npy'))
            np.savez_compressed(f/f'{domain}_boundary_signal.npz',top1=top1,top2=top2,margin=margin,
                pair_low=low,pair_high=high,unordered_counts=counts,frequency_ranking=ranking)
            for j,(a,b) in enumerate(PAIRS):
                mask=(low==a)&(high==b);n=int(mask.sum())
                allpressure.append({'seed':seed,'domain':domain,'pair':f'C{a+1}/C{b+1}',
                    'frequency_rank':int(np.flatnonzero(ranking==j)[0]+1),'n':n,'fraction_percent':100*n/len(logits),
                    'median_top1_top2_margin':float(np.median(margin[mask])) if n else None})
        t=np.load(f/'target_boundary_signal.npz')
        directed=[]
        for a in range(7):
            for b in range(7):
                if a==b:continue
                m=(t['top1']==a)&(t['top2']==b);n=int(m.sum());den=int((t['top1']==a).sum())
                directed.append({'seed':seed,'top1':f'C{a+1}','top2':f'C{b+1}','n':n,
                    'fraction_all_percent':100*n/len(m),'fraction_given_top1_percent':100*n/den if den else None,
                    'median_margin':float(np.median(t['margin'][m])) if n else None})
        csvwrite(f'directed_frequency_seed_{seed}.csv',directed)
    csvwrite('unlabeled_pair_frequency.csv',allpressure);dump(OUT/'provenance.json',prov)
    frozen={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'ALL_BOUNDARY_SIGNALS_FROZEN.json',{'all_seeds_completed':True,'target_GT_opened':False,
        'all21_pairs_and_prefix_sizes1_to21_prespecified':True,'no_training_or_boundary_selection':True,'files_sha256':frozen})
    for rel,h in frozen.items():assert sha(OUT/rel)==h
    print('All unlabeled boundary signals/rankings hash-frozen; now opening target GT offline.',flush=True)
    y=np.load(DATA/'target_labels.npy');gt=np.load(DATA/'target_gt.npy',mmap_mode='r')
    assert hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()==old['historical_metadata']['domains']['target']['labels_sha256']
    coords=np.load(OUT/'seed_2100/target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
    pairs,coverage,summaryrows,subsets=[],[],[],[]
    for seed in SEEDS:
        t=np.load(OUT/f'seed_{seed}/target_boundary_signal.npz');p=t['top1'];second=t['top2'];errors=p!=y;local=errors&(second==y)
        truthlow=np.minimum(y,p);truthhigh=np.maximum(y,p)
        errorcounts=np.array([(errors&(truthlow==a)&(truthhigh==b)).sum() for a,b in PAIRS])
        counts=t['unordered_counts'];ranks=t['frequency_ranking']
        rho=float(spearmanr(counts,errorcounts).statistic)
        summaryrows.append({'seed':seed,'R0_OA':100*float((~errors).mean()),'n_errors':int(errors.sum()),
            'true_class_top2_on_errors_percent':100*float(local[errors].mean()),
            'top2_oracle_OA':100*float(((~errors)|local).mean()),'pair_frequency_vs_GT_error_mass_Spearman':rho})
        for j,(a,b) in enumerate(PAIRS):
            comp=(t['pair_low']==a)&(t['pair_high']==b);trueedge=errors&(truthlow==a)&(truthhigh==b)
            pairs.append({'seed':seed,'pair':f'C{a+1}/C{b+1}','unlabeled_frequency_rank':int(np.flatnonzero(ranks==j)[0]+1),
                'competition_n':int(counts[j]),'competition_percent':100*counts[j]/len(y),
                'GT_confusion_error_n':int(errorcounts[j]),'GT_confusion_error_share_percent':100*errorcounts[j]/errors.sum(),
                'same_sample_true_boundary_and_competition_n':int((trueedge&comp).sum()),
                'competition_sample_error_percent':100*float(errors[comp].mean()) if comp.any() else None,
                'GT_edge_sample_competition_match_percent':100*float(comp[trueedge].mean()) if trueedge.any() else None})
        for m in range(1,22):
            selected=np.zeros(21,bool);selected[ranks[:m]]=True
            gtcovered=np.zeros(len(y),bool);compcovered=np.zeros(len(y),bool)
            for j,(a,b) in enumerate(PAIRS):
                if selected[j]:
                    gtcovered|=errors&(truthlow==a)&(truthhigh==b)
                    compcovered|=(t['pair_low']==a)&(t['pair_high']==b)
            coverage.append({'seed':seed,'prefix_size':m,'pair_list':';'.join(f'C{PAIRS[j][0]+1}/C{PAIRS[j][1]+1}' for j in ranks[:m]),
                'competition_mass_percent':100*float(compcovered.mean()),
                'aggregate_true_error_edge_coverage_percent':100*gtcovered.sum()/errors.sum(),
                'same_sample_local_correct_boundary_coverage_percent':100*(compcovered&local).sum()/errors.sum(),
                'uniform_random_pair_set_expected_edge_coverage_percent':100*m/21})
        masks={'all_errors':errors}
        for c in range(7):masks[f'C{c+1}_errors']=errors&(y==c)
        for a,b in [(6,5),(0,6),(6,0)]:masks[f'C{a+1}_to_C{b+1}']=errors&(y==a)&(p==b)
        for name,mask in masks.items():
            if mask.any():subsets.append({'seed':seed,'subset':name,'n':int(mask.sum()),'true_class_equals_top2_percent':100*float((second[mask]==y[mask]).mean())})
    csvwrite('summary_per_seed.csv',summaryrows);csvwrite('pair_correspondence_per_seed.csv',pairs)
    csvwrite('frequency_prefix_coverage_per_seed.csv',coverage);csvwrite('local_top2_coverage_per_class_and_confusion.csv',subsets)
    sa=aggregate(summaryrows,[],['R0_OA','true_class_top2_on_errors_percent','top2_oracle_OA','pair_frequency_vs_GT_error_mass_Spearman'])
    ca=aggregate(coverage,['prefix_size'],['competition_mass_percent','aggregate_true_error_edge_coverage_percent','same_sample_local_correct_boundary_coverage_percent'])
    pa=aggregate(pairs,['pair'],['unlabeled_frequency_rank','competition_percent','GT_confusion_error_share_percent','competition_sample_error_percent','GT_edge_sample_competition_match_percent'])
    csvwrite('prefix_coverage_seed_aggregate.csv',ca);csvwrite('pair_correspondence_seed_aggregate.csv',pa)
    output={'summary':sa,'prefix_coverage':ca,'pair_correspondence':pa,'target_GT_scoring_only_after_freeze':True,
        'no_training_threshold_or_boundary_selection':True,'limits':[
            'Top1/top2 are model predictions, but no target predicted class is used as a supervision label.',
            'Frequency means competition activity, not proven uncertainty/danger or correct direction of boundary movement.',
            'All21prefix sizes retained; GT not used to select a boundary subset or subsequent method.',
            'Aggregate edge coverage and per-sample true class in top2 are distinct; this diagnostic does not establish training benefit.']}
    dump(OUT/'summary.json',output)
    report(output,allpressure)
    for rel,h in frozen.items():assert sha(OUT/rel)==h
    for rec in old['seeds'].values():assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    dump(OUT/'SCORING_COMPLETE.json',{'unlabeled_signals_and_checkpoints_unchanged':True,
       'no_training_boundary_selection_or_automatic_followup':True,
       'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='SCORING_COMPLETE.json'}})

def report(s,pressure):
    lines=['Houston target aggregate boundary-signal diagnostic; originalC3seeds epoch200.',
        'Unlabeled top1/top2 statistics frozen before GT scoring; no training/boundary-subset selection.',
        'Frequency/activity and true error edges are different; all21 frequency-prefix sizes reported.',
        'Mean +/- sampleSD across3modelseeds.','']
    for key,val in s['summary'][0].items():lines.append(f'{key}: {val}')
    lines.append('\nUnlabeled top10 pairs perseed (not selected for any method):')
    for seed in SEEDS:
        rows=sorted([r for r in pressure if r['seed']==seed and r['domain']=='target'],key=lambda r:r['frequency_rank'])
        lines.append(str(seed)+': '+'; '.join(f"{r['pair']} {r['fraction_percent']:.2f}% (median margin{r['median_top1_top2_margin']:.3f})" for r in rows[:10]))
    lines.append('\nAll prefixes:')
    for r in s['prefix_coverage']:lines.append(f"prefix{r['prefix_size']}: competition mass={r['competition_mass_percent_mean']:.2f} +/- {r['competition_mass_percent_std']:.2f}%; aggregate true-error edge coverage={r['aggregate_true_error_edge_coverage_percent_mean']:.2f} +/- {r['aggregate_true_error_edge_coverage_percent_std']:.2f}%; local same-sample true boundary coverage={r['same_sample_local_correct_boundary_coverage_percent_mean']:.2f} +/- {r['same_sample_local_correct_boundary_coverage_percent_std']:.2f}%")
    lines.extend(['\nLimits:',*s['limits']]);(OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n');(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True);rs=s['prefix_coverage'];m=[r['prefix_size'] for r in rs]
    for key,label in [('aggregate_true_error_edge_coverage_percent','Aggregate GT error-edge coverage'),('same_sample_local_correct_boundary_coverage_percent','Same-sample true class is top2'),('competition_mass_percent','Unlabeled competition mass')]:
        ax.errorbar(m,[r[key+'_mean'] for r in rs],yerr=[r[key+'_std'] for r in rs],capsize=2,label=label)
    ax.plot(m,100*np.array(m)/21,'--',color='grey',label='Uniform random edge-set expectation')
    ax.set(xlabel='All prespecified frequency-ranked prefix sizes',ylabel='Samples / errors covered (%)',ylim=(0,101),xticks=range(1,22,2),title='Frozen aggregate boundary activity vs GT confusion;3seed mean +/- SD');ax.legend(fontsize=8)
    fig.savefig(OUT/'boundary_signal_coverage.png',dpi=180);plt.close(fig)
    print('\n'.join(lines[:20]),flush=True)

if __name__=='__main__':main()
