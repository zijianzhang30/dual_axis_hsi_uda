"""Report/plot existing diagnostic outputs only; does not search or fit anything."""
import csv
import json
import numpy as np
from common import OUT, HERE, SEEDS, STAGES, KS, sha, dump

def main():
    completion=json.loads((OUT/'SCORING_COMPLETE.json').read_text())
    assert sha(OUT/'summary.json')==completion['summary_sha256']
    assert sha(OUT/'REPORT.txt')==completion['report_sha256']
    summary=json.loads((OUT/'summary.json').read_text());runs=summary['runs']
    def values(stage,k,subset,key):
        return np.array([runs[str(s)]['stages'][stage][str(k)]['subsets'][subset][key] for s in SEEDS],float)
    def mean(stage,k,subset,key):return float(values(stage,k,subset,key).mean())
    # Preserve unweighted class macro separately from pixel-micro subset aggregates.
    rows=[]
    for stage in STAGES:
        for k in KS:
            for subset in ['all','correct','error']:
                rr=[r for r in summary['macro_rows'] if r['stage']==stage and r['k']==k and r['subset']==subset]
                for key in rr[0]:
                    if key in ['seed','stage','k','subset']:continue
                    vals=[r[key] for r in rr if r[key] is not None]
                    if vals:rows.append({'stage':stage,'k':k,'subset':subset,'metric':key,'mean':float(np.mean(vals)),
                                         'sample_std':float(np.std(vals,ddof=1)),'values':vals})
    with (OUT/'macro_seed_aggregate.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['Houston local support V1：冻结离线诊断结论。没有训练、调参或后续实验。',
           'T/Z、原分类器 logits/predictions 和精确邻居均在本诊断读取 target GT 前独立冻结。',
           '下述百分比均为三个优化 seed 的均值；错误子集由固定原 Z 分类器定义，T/Z 使用同一批样本。',
           '支持率与 purity 是真实类邻居票数/k，不是模型置信度。','',
           '1. 目标内部近邻同类，跨域 source 同类近邻却缺失，是大量错误样本的共同现象。']
    for k in KS:
        lines.append(f'k={k}: 错误样本 source 正确类支持 T={100*mean("T",k,"error","source_correct_support_mean"):.2f}% '
                     f'→ Z={100*mean("Z",k,"error","source_correct_support_mean"):.2f}%；target purity '
                     f'T={100*mean("T",k,"error","target_true_class_purity_mean"):.2f}% '
                     f'→ Z={100*mean("Z",k,"error","target_true_class_purity_mean"):.2f}%。')
    lines.extend(['','2. T→Z 对最终正确与错误样本的影响不同，不能只看全样本平均。'])
    for k in KS:
        lines.append(f'k={k}: 最终正确样本 source 正确支持 '
                     f'{100*mean("T",k,"correct","source_correct_support_mean"):.2f}% '
                     f'→ {100*mean("Z",k,"correct","source_correct_support_mean"):.2f}%；'
                     f'最终错误样本则按上述结果下降。')
    lines.extend(['','3. Z 错误样本的归类：'])
    for k in KS:
        lines.append(f'k={k}: source 支持不足 {100*mean("Z",k,"error","source_local_support_error_fraction"):.2f}%；'
                     f'target 局部混合（独立标记、可重叠）{100*mean("Z",k,"error","target_local_mixing_fraction"):.2f}%；'
                     f'两侧均有严格正确类局部支持但分类器错误 {100*mean("Z",k,"error","classifier_boundary_disagreement_fraction"):.2f}%。')
        lines.append('互斥归类采用 source 优先：'+', '.join(f'{summary["category_labels"][i]} '
                     f'{100*mean("Z",k,"error",f"category_{i}_fraction_of_error"):.2f}%' for i in range(3)))
    lines.extend(['','4. C1/C7 不宜用一个原因概括；所有非空混淆对已有独立统计。'])
    for pair in ['C1_to_C2','C1_to_C3','C1_to_C7','C7_to_C6','C7_to_C5','C6_to_C5']:
        for k in KS:
            if all(pair in runs[str(s)]['stages']['T'][str(k)]['subsets'] for s in SEEDS):
                lines.append(f'{pair} k{k}: source 支持 '
                             f'{100*mean("T",k,pair,"source_correct_support_mean"):.2f}% '
                             f'→ {100*mean("Z",k,pair,"source_correct_support_mean"):.2f}%；target purity '
                             f'{100*mean("T",k,pair,"target_true_class_purity_mean"):.2f}% '
                             f'→ {100*mean("Z",k,pair,"target_true_class_purity_mean"):.2f}%。')
    lines.extend(['','解释边界：',
                  '类质心对应正确与样本的 source 局部支持不足可以同时成立；本轮细化了上一轮质心诊断。',
                  '分类器边界不一致占比低，只排除了本协议的“两侧严格正确支持仍被错分”现象占主导；',
                  '不能据此证明修改 classifier/loss 无效，也不能证明 Transformer 造成错误。',
                  '子集按最终 Z 分类器结果划分，属于结果条件化的描述；不能视为独立因果对照。',
                  '原 logit margin 在 T/Z 共用，变化为零；没有另行构造 T 分类器。',
                  'target LOO 仅排除自身，空间邻接/patch 重叠可提高 purity；source 类比例也影响支持率。',
                  '固定 k=5/15 均完整报告，没有选择更有利的 k。独立标记重叠，互斥归类受既定优先级影响。',
                  '一个场景对、三个优化种子，未作显著性或跨场景普适性声明。'])
    (OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n')
    plot(values,summary)
    dump(OUT/'FINAL_DELIVERY.json',{'scoring_complete_sha256':sha(OUT/'SCORING_COMPLETE.json'),
         'findings_sha256':sha(OUT/'FINDINGS.txt'),'macro_aggregate_sha256':sha(OUT/'macro_seed_aggregate.csv'),
         'plot_sha256':sha(OUT/'local_support_signals.png'),'reporting_code_sha256':sha(HERE/'summarize.py'),
         'reporting_only_no_new_experiment':True})
    print('\n'.join(lines[:16]))

def plot(values,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    labels=['T k5','Z k5','T k15','Z k15'];combos=[('T',5),('Z',5),('T',15),('Z',15)]
    xx=np.arange(4);width=.35
    for offset,subset,color in [(-width/2,'correct','#2a9d8f'),(width/2,'error','#e76f51')]:
        a=np.array([values(st,k,subset,'source_correct_support_mean') for st,k in combos])*100
        axes[0].bar(xx+offset,a.mean(1),width,label=f'Final {subset}',color=color,yerr=a.std(1,ddof=1),capsize=3)
    axes[0].set(title='Correct source-class support',ylabel='Neighbor votes (%)',xticks=xx,xticklabels=labels,ylim=(0,105))
    axes[0].legend()
    a=np.array([values(st,k,'error','target_true_class_purity_mean') for st,k in combos])*100
    axes[1].bar(xx,a.mean(1),color=['#457b9d','#e9c46a']*2,yerr=a.std(1,ddof=1),capsize=3)
    axes[1].set(title='Target purity on final errors',ylabel='LOO true-class votes (%)',xticks=xx,xticklabels=labels,ylim=(0,105))
    bottom=np.zeros(4)
    for i,color in enumerate(['#e76f51','#e9c46a','#2a9d8f']):
        a=np.array([values(st,k,'error',f'category_{i}_fraction_of_error') for st,k in combos])*100
        axes[2].bar(xx,a.mean(1),bottom=bottom,label=summary['category_labels'][i],color=color)
        bottom+=a.mean(1)
    axes[2].set(title='Exclusive final-error categories',ylabel='Errors (%)',xticks=xx,xticklabels=labels,ylim=(0,105))
    axes[2].legend(loc='lower left',fontsize=8)
    fig.suptitle('Houston Joint BN+C, fixed epoch200; mean of 3 seeds. Source-first partition; descriptive only.')
    fig.savefig(OUT/'local_support_signals.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
