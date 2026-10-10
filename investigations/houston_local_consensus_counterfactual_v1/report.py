"""Summarize the complete frozen grid; never selects a counterfactual configuration."""
import csv
import json
import numpy as np
from common import HERE,OUT,SEEDS,STAGES,MODES,KS,sha,dump

def main():
    done=json.loads((OUT/'SCORING_COMPLETE.json').read_text())
    assert sha(OUT/'summary.json')==done['summary_sha256']
    summary=json.loads((OUT/'summary.json').read_text());rows=summary['per_seed_rows']
    lookup={(r['seed'],r['stage'],r['mode'],r['k_target'],r['k_source'],r['subset']):r for r in rows}
    def vals(stage,mode,kt,ks,subset,key):
        return np.array([lookup[(s,stage,mode,kt,ks,subset)][key] for s in SEEDS
                         if (s,stage,mode,kt,ks,subset) in lookup and lookup[(s,stage,mode,kt,ks,subset)][key] is not None],float)
    def fmt(a,percent=False):
        factor=100 if percent else 1
        a=np.asarray(a)*factor
        return f'{a.mean():.4f} +/- {a.std(ddof=1):.4f}' if len(a)>1 else (f'{a[0]:.4f} (one seed)' if len(a) else 'NA')
    lines=['Houston local consensus counterfactual V1; frozen Joint BN+C original C, seeds2100/2101/2102, epoch200.',
           'No training/backward/tuning/follow-up. Complete fixed grid: stageT/Z, standard/nonoverlap, targetk5/15, sourcek5/15.',
           'Strict authenticated cache reuse. Float64 unit features; consensus=unit(q+mean(original unit target neighbors)).',
           'Source references2393 original train samples; target52901. Source labels construct predictions before target GT.',
           'ALL_COUNTERFACTUALS_FROZEN.json covers every feature, neighbor, prediction and audit before target label scoring.',
           'Nonoverlap control: reliable center coordinates, exclude Chebyshev<=12; verified every retained neighbor>12.',
           'No counterfactual original classifier: native raw-CLS normalization differs from unit consensus.',
           'Native classifier defines common original-correct/error/confusion subsets; its logits are unchanged.',
           'Below means +/- sample SD across available optimization seeds. Support/accuracy fractions displayed in percent.',
           'Recovery denominator is explicitly stated; counts sum the same pixel pool repeated across seeds, not independent pixels.','']
    for subset in ['all','original_correct','original_error','C1_original_error','C7_original_error','C7_to_C6']:
        lines.append(subset)
        for stage in STAGES:
            for mode in MODES:
                for kt in KS:
                    for ks in KS:
                        get=lambda key:vals(stage,mode,kt,ks,subset,key)
                        if not len(get('n')):continue
                        lines.append(f'  {stage} {mode} kt{kt} ks{ks}; n={get("n").astype(int).tolist()}')
                        lines.append('    source support before '+fmt(get('source_correct_support_before'),True)+' -> after '+fmt(get('source_correct_support_after'),True)+
                                     '; paired delta '+fmt(get('source_correct_support_delta'),True)+' pp')
                        lines.append('    source support margin before '+fmt(get('source_support_margin_before'))+' -> after '+fmt(get('source_support_margin_after'))+
                                     '; paired delta '+fmt(get('source_support_margin_delta')))
                        lines.append('    support restored n='+str(get('source_support_restored_n').astype(int).tolist())+
                                     '; %subset='+fmt(get('source_support_restored_fraction'),True)+
                                     '; %before-insufficient='+fmt(get('source_support_restored_fraction_of_before_insufficient'),True)+
                                     '; support lost n='+str(get('source_support_lost_n').astype(int).tolist()))
                        for kind in ['source_knn','source_centroid']:
                            lines.append(f'    {kind}: accuracy '+fmt(get(kind+'_accuracy_before'),True)+' -> '+fmt(get(kind+'_accuracy_after'),True)+
                                         '; paired delta '+fmt(get(kind+'_accuracy_delta'),True)+' pp; wrong->correct n='+
                                         str(get(kind+'_wrong_to_correct_n').astype(int).tolist())+'; correct->wrong n='+
                                         str(get(kind+'_correct_to_wrong_n').astype(int).tolist()))
                        lines.append('    target neighborhood purity='+fmt(get('target_neighbor_purity'),True)+
                                     '; overlapping neighbor fraction='+fmt(get('target_neighbor_overlap_fraction'),True))
        lines.append('')
    lines.extend(['Limitations:',*summary['limitations']])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    findings=['Houston local consensus counterfactual V1：完整固定网格的描述性结果。',
              '原C三seed/epoch200；无训练、反传、调参或后续实验。所有 counterfactual features/neighbors/predictions 先冻结，再读取 target GT。',
              'standard 与 nonoverlap、target/source k5/15、T/Z 全部保留，不选择最好设置。',
              '所有下述百分比为三seed均值，误差为sample SD；变化按同seed相减。','',
              '原分类器错误样本：source正确类支持 before→consensus，以及支持/预测恢复。']
    compact=[]
    for stage in STAGES:
        for mode in MODES:
            for kt in KS:
                for ks in KS:
                    get=lambda key:vals(stage,mode,kt,ks,'original_error',key)
                    findings.append(f'{stage} {mode} targetk{kt}/sourcek{ks}: support '
                        f'{fmt(get("source_correct_support_before"),True)} → {fmt(get("source_correct_support_after"),True)}%；'
                        f'同seedΔ={fmt(get("source_correct_support_delta"),True)}pp；'
                        f'恢复严格支持占原错误 {fmt(get("source_support_restored_fraction"),True)}%；'
                        f'knn错→对占原错误 {fmt(get("source_knn_wrong_to_correct_fraction"),True)}%。')
                    compact.append({'stage':stage,'mode':mode,'k_target':kt,'k_source':ks,
                         'error_support_before_percent':float(100*get('source_correct_support_before').mean()),
                         'error_support_after_percent':float(100*get('source_correct_support_after').mean()),
                         'error_support_paired_delta_pp':float(100*get('source_correct_support_delta').mean()),
                         'error_support_recovery_percent':float(100*get('source_support_restored_fraction').mean()),
                         'error_knn_wrong_to_correct_percent':float(100*get('source_knn_wrong_to_correct_fraction').mean()),
                         'original_correct_knn_damage_percent':float(100*vals(stage,mode,kt,ks,'original_correct','source_knn_correct_to_wrong_fraction').mean())})
    findings.extend(['','原分类器正确样本的破坏：'])
    for stage in STAGES:
        for mode in MODES:
            for kt in KS:
                for ks in KS:
                    findings.append(f'{stage} {mode} targetk{kt}/sourcek{ks}: source-knn原对→错占原分类器正确样本 '
                        f'{fmt(vals(stage,mode,kt,ks,"original_correct","source_knn_correct_to_wrong_fraction"),True)}%；'
                        f'严格支持丢失 {fmt(vals(stage,mode,kt,ks,"original_correct","source_support_lost_fraction"),True)}%。')
    findings.extend(['','neighbor_overlap_audit：原分类器错误样本，邻域purity和source支持恢复的配对对照。'])
    for stage in STAGES:
        for kt in KS:
            ps=vals(stage,'standard_knn',kt,5,'original_error','target_neighbor_purity')
            pn=vals(stage,'nonoverlap_knn',kt,5,'original_error','target_neighbor_purity')
            findings.append(f'{stage} targetk{kt}: standard purity {fmt(ps,True)}%；nonoverlap purity {fmt(pn,True)}%；'
                f'同seed nonoverlap-standard={fmt(pn-ps,True)}pp；standard重叠邻居 '
                f'{fmt(vals(stage,"standard_knn",kt,5,"original_error","target_neighbor_overlap_fraction"),True)}%，nonoverlap为0%。')
            for ks in KS:
                rs=vals(stage,'standard_knn',kt,ks,'original_error','source_support_restored_fraction')
                rn=vals(stage,'nonoverlap_knn',kt,ks,'original_error','source_support_restored_fraction')
                findings.append(f'  sourcek{ks}: 支持恢复nonoverlap-standard={fmt(rn-rs,True)}pp。')
    findings.extend(['','解释边界：',
        '恢复严格支持：before正确类票数<=最强错误票数，after严格大于；分母同时报告原错误总数和原支持不足数。',
        'source-knn错→对按该readout before/after判定；原分类器正确/错误仅定义固定分组，不能混用两种正确性。',
        '原正确样本破坏为source-knn before正确、consensus错误；不能称原模型分类器被改变。',
        '非重叠排除了共享patch像素，但不排除空间自相关；purity下降不能单独证明此前高purity主要由重叠造成。',
        'source类组成影响邻居票数；局部consensus不保证提供缺失的跨域类别对应。',
        '没有调用原classifier，readout变化不等同于原模型OA变化；所有结果仅为冻结表示的counterfactual诊断。'])
    (OUT/'FINDINGS.txt').write_text('\n'.join(findings)+'\n')
    with (OUT/'compact_findings.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
    # Three-seed aggregates of the class-macro endpoints, separate from sample micro.
    macro=[]
    for stage in STAGES:
        for mode in MODES:
            for kt in KS:
                for ks in KS:
                    for subset in ['all','original_correct','original_error']:
                        rr=[r for r in summary['class_macro_rows'] if r['stage']==stage and r['mode']==mode and r['k_target']==kt and r['k_source']==ks and r['subset']==subset]
                        for metric in rr[0]:
                            if metric in ['seed','stage','mode','k_target','k_source','subset']:continue
                            v=[r[metric] for r in rr if r[metric] is not None]
                            if v:macro.append({'stage':stage,'mode':mode,'k_target':kt,'k_source':ks,'subset':subset,
                                'metric':metric,'mean':float(np.mean(v)),'sample_std':float(np.std(v,ddof=1)),'values':v})
    with (OUT/'class_macro_seed_aggregate.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(macro[0]));w.writeheader();w.writerows(macro)
    plot(vals)
    dump(OUT/'FINAL_DELIVERY.json',{'no_training_backward_tuning_followup':True,
        'scoring_complete_sha256':sha(OUT/'SCORING_COMPLETE.json'),'report_sha256':sha(OUT/'REPORT.txt'),
        'findings_sha256':sha(OUT/'FINDINGS.txt'),'figure_sha256':sha(OUT/'before_consensus_source_support.png'),
        'reporting_code_sha256':sha(HERE/'report.py'),'complete_fixed_grid_reported':True})
    print('REPORT/FINDINGS/CSV/figure saved; entire fixed grid retained',flush=True)

def plot(vals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    combos=[('standard_knn',5),('standard_knn',15),('nonoverlap_knn',5),('nonoverlap_knn',15)]
    labels=['Standard\nkt=5','Standard\nkt=15','Nonoverlap\nkt=5','Nonoverlap\nkt=15']
    x=np.arange(4);width=.36
    for i,stage in enumerate(STAGES):
        for j,ks in enumerate(KS):
            ax=axes[i,j]
            before=np.array([vals(stage,mode,kt,ks,'original_error','source_correct_support_before') for mode,kt in combos])*100
            after=np.array([vals(stage,mode,kt,ks,'original_error','source_correct_support_after') for mode,kt in combos])*100
            ax.bar(x-width/2,before.mean(1),width,color='#999999',label='Before',yerr=before.std(1,ddof=1),capsize=3)
            ax.bar(x+width/2,after.mean(1),width,color=['#2a9d8f','#2a9d8f','#e9c46a','#e9c46a'],label='Consensus',yerr=after.std(1,ddof=1),capsize=3)
            ax.set(title=f'{stage}, source k={ks}',ylabel='Correct source-class votes (%)',xticks=x,xticklabels=labels)
            ax.legend()
    fig.suptitle('Fixed original-classifier errors: before -> consensus support\nMean +/- sample SD, 3 seeds; target/source k5/15 all retained')
    fig.savefig(OUT/'before_consensus_source_support.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
