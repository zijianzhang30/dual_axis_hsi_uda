"""Report every predeclared oracle transform without selecting any method."""
import csv
import json
import numpy as np
from common import HERE,OUT,SEEDS,GROUPS,KS,sha,dump

def main():
    gate=json.loads((OUT/'ORACLE_RESULTS_FROZEN.json').read_text())
    assert gate['native_inputs_frozen_before_target_GT'] and gate['target_GT_used_for_oracle_G2_G3']
    assert sha(OUT/'summary.json')==gate['files_sha256']['summary.json']
    summary=json.loads((OUT/'summary.json').read_text());metrics=summary['metrics'];rows=summary['subsets']
    def values(group,k,metric):
        return np.array([r[metric] for r in metrics if r['group']==group and r['k_source']==k],float)
    def subset_values(group,k,subset,key):
        return np.array([r[key] for r in rows if r['group']==group and r['k_source']==k and r['subset']==subset and r[key] is not None],float)
    def fmt(a,percent=False):
        a=np.asarray(a)*(100 if percent else 1)
        return f'{a.mean():.4f} +/- {a.std(ddof=1):.4f}' if len(a)>1 else (f'{a[0]:.4f}' if len(a) else 'NA')
    lines=['Houston oracle transport diagnostic V1: original C Joint BN+C, seeds2100/2101/2102, epoch200.',
        'Raw native CLS/head authenticated and hash-frozen before target GT. Native G0 classifier argmax exactly replays.',
        'No training/backward/tuning/checkpoint selection/follow-up.',
        'G0 identity; G1 global shared Gaussian moment affine; G2 oracle class centroid translation;',
        'G3 oracle class-wise Gaussian moment affine. G3 is not orthogonal and is separate from G2.',
        'Transport rawZ64; frozen native head reads raw transformedZ; source cosine kNN normalizes separately.',
        'Source2393 original train; target52901. Affine ridge fixed1e-6*trace(C)/64, no search.',
        'G2/G3 use GT for routing and same-target-sample moment estimation; oracle results are optimistic.',
        'All below mean +/- sample SD of3 optimization seeds; all k/transforms reported without selection.','']
    for group in GROUPS:
        lines.append(group)
        lines.append('  frozen classifier OA='+fmt(values(group,5,'frozen_classifier_OA'))+'%; AA='+fmt(values(group,5,'frozen_classifier_AA'))+'%')
        for k in KS:
            lines.append(f'  sourcek{k}: OA='+fmt(values(group,k,'source_knn_OA'))+'%; AA='+fmt(values(group,k,'source_knn_AA'))+'%; '
                         'support micro='+fmt(values(group,k,'source_correct_support_micro'),True)+'%; macro='+fmt(values(group,k,'source_correct_support_macro'),True)+'%')
            lines.append('    source margin micro='+fmt(values(group,k,'source_support_margin_micro'))+'; macro='+fmt(values(group,k,'source_support_margin_macro')))
    for subset in ['original_error','original_correct','C1_all','C7_all','C1_original_error','C7_original_error','C7_to_C6']:
        lines.extend(['',subset])
        for group in GROUPS:
            lines.append(group+' classifier accuracy='+fmt(subset_values(group,5,subset,'classifier_accuracy_after'),True)+'%; '
                         'restored n='+str(subset_values(group,5,subset,'classifier_recovered_n').astype(int).tolist())+'; '
                         'damaged n='+str(subset_values(group,5,subset,'classifier_damaged_n').astype(int).tolist()))
            for k in KS:
                lines.append(f'  sourcek{k}: support='+fmt(subset_values(group,k,subset,'source_support_after'),True)+'%; '
                    'margin='+fmt(subset_values(group,k,subset,'source_margin_after'))+'; '
                    'knn accuracy='+fmt(subset_values(group,k,subset,'source_knn_accuracy_after'),True)+'%; '
                    'knn recovered n='+str(subset_values(group,k,subset,'source_knn_recovered_n').astype(int).tolist())+'; '
                    'knn damaged n='+str(subset_values(group,k,subset,'source_knn_damaged_n').astype(int).tolist()))
    lines.extend(['','Transform audits (regularized covariance residual/condition/rank):'])
    for seed in SEEDS:
        audit=summary['transform_audits'][str(seed)]
        lines.append(f'seed{seed} G1 '+str(audit['G1']))
        for c,r in audit['G3'].items():lines.append(f'  G3 C{c} '+str(r))
    lines.extend(['','Limitations:',*summary['limitations']])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    findings=['Houston oracle transport V1：全部固定变换的机制诊断。没有训练、反传、调参或后续实验。',
        '原C三seed、epoch200；raw CLS Z、原logits/predictions和冻结分类器head先校验/冻结，再读取target GT。',
        'G1与G3采用同一固定affine moment transport公式；G2仅平移质心。k5/15全部保留。',
        'G2/G3明确使用真类路由及同一评估集的类别统计，只是oracle analysis，不能视为可部署UDA结果。','']
    for group in GROUPS:
        findings.append(f'{group}: 冻结原classifier OA={fmt(values(group,5,"frozen_classifier_OA"))}%，AA={fmt(values(group,5,"frozen_classifier_AA"))}%。')
        for k in KS:
            findings.append(f'  source-kNN k{k}: OA={fmt(values(group,k,"source_knn_OA"))}%，AA={fmt(values(group,k,"source_knn_AA"))}%；'
                            f'正确类支持micro={fmt(values(group,k,"source_correct_support_micro"),True)}%，'
                            f'macro={fmt(values(group,k,"source_correct_support_macro"),True)}%。')
    findings.extend(['','原错误恢复与原正确破坏（冻结原classifier）：'])
    for group in GROUPS:
        findings.append(f'{group}: 原错误恢复 {fmt(subset_values(group,5,"original_error","classifier_recovered_fraction_of_subset"),True)}%；'
                        f'原正确破坏 {fmt(subset_values(group,5,"original_correct","classifier_damaged_fraction_of_subset"),True)}%。')
    findings.extend(['','重点类别/混淆对的冻结classifier正确率：'])
    for subset in ['C1_all','C7_all','C7_to_C6']:
        findings.append(subset+': '+'; '.join(f'{g}={fmt(subset_values(g,5,subset,"classifier_accuracy_after"),True)}%' for g in GROUPS))
    findings.extend(['','预先固定的解释方式：',
        'G1对G0：检验这一全局均值/协方差变换能修复多少误差；差结果可能受到源目标类别比例不同影响。',
        'G2对G0/G1：检验真类均值偏移校正的修复幅度；真类路由本身使用了答案，因此不能据此证明可识别的class-conditional适应。',
        'G3对G2：检验加入类内协方差形状校正后的额外修复；G3并非任意local-manifold变换。',
        'G3后残余误差可与更复杂的局部/非线性/多模态变化相容，但不能直接证明这些机制。',
        '同批target统计存在自包含乐观性；C4只有22样本、协方差低秩，采用预先固定数值ridge并完整审计。',
        '没有用结果选择checkpoint、k、ridge、transform或正式方法。'])
    (OUT/'FINDINGS.txt').write_text('\n'.join(findings)+'\n')
    # Aggregated per-class/subset endpoints (counts and rates retained separately).
    aggregate=[]
    for group in GROUPS:
        for k in KS:
            for subset in sorted({r['subset'] for r in rows if r['group']==group and r['k_source']==k}):
                rr=[r for r in rows if r['group']==group and r['k_source']==k and r['subset']==subset]
                for metric in rr[0]:
                    if metric in ['seed','group','k_source','subset']:continue
                    vals=[r[metric] for r in rr if r[metric] is not None]
                    if vals:aggregate.append({'group':group,'k_source':k,'subset':subset,'metric':metric,'available_seeds':len(vals),
                        'mean':float(np.mean(vals)),'sample_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'values':vals})
    with (OUT/'subset_seed_aggregate.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    plot(values,subset_values)
    dump(OUT/'FINAL_DELIVERY.json',{'report_sha256':sha(OUT/'REPORT.txt'),'findings_sha256':sha(OUT/'FINDINGS.txt'),
        'figure_sha256':sha(OUT/'oracle_transport_summary.png'),'reporting_code_sha256':sha(HERE/'report.py'),
        'input_freeze_sha256':sha(OUT/'ALL_INPUTS_FROZEN.json'),'oracle_freeze_sha256':sha(OUT/'ORACLE_RESULTS_FROZEN.json'),
        'no_training_backward_tuning_followup':True})
    print('\n'.join(findings[:17]),flush=True)

def plot(values,subset_values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True);x=np.arange(4);width=.25
    for off,key,label,color,k in [(-width,'frozen_classifier_OA','Frozen classifier','#457b9d',5),
                                (0,'source_knn_OA','Source kNN5','#2a9d8f',5),
                                (width,'source_knn_OA','Source kNN15','#e9c46a',15)]:
        a=np.array([values(g,k,key) for g in GROUPS]);axes[0].bar(x+off,a.mean(1),width,yerr=a.std(1,ddof=1),capsize=3,label=label,color=color)
    axes[0].set(title='All-target OA',ylabel='Accuracy (%)',xticks=x,xticklabels=GROUPS,ylim=(0,105));axes[0].legend(fontsize=8)
    for k,color,off in [(5,'#2a9d8f',-.18),(15,'#e9c46a',.18)]:
        a=np.array([values(g,k,'source_correct_support_micro') for g in GROUPS])*100
        axes[1].bar(x+off,a.mean(1),.36,yerr=a.std(1,ddof=1),capsize=3,label=f'Source k{k}',color=color)
    axes[1].set(title='Source correct-class support',ylabel='Votes (%)',xticks=x,xticklabels=GROUPS,ylim=(0,105));axes[1].legend()
    for off,subset,label,color in [(-.18,'C1_all','C1','#457b9d'),(.18,'C7_all','C7','#e76f51')]:
        a=np.array([subset_values(g,5,subset,'classifier_accuracy_after') for g in GROUPS])*100
        axes[2].bar(x+off,a.mean(1),.36,yerr=a.std(1,ddof=1),capsize=3,label=label,color=color)
    axes[2].set(title='Frozen classifier class recall',ylabel='Recall (%)',xticks=x,xticklabels=GROUPS,ylim=(0,105));axes[2].legend()
    fig.suptitle('Houston raw CLS oracle transport; 3 seeds mean +/- SD\nG2/G3 use true-class routing and same-target-set moments; diagnostic only')
    fig.savefig(OUT/'oracle_transport_summary.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
