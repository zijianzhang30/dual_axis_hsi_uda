"""Complete fixed-variant reporting; no selection of a successful variant."""
import csv
import json
import numpy as np
from common import HERE,OUT,GROUPS,KS,sha,dump

def main():
    done=json.loads((OUT/'SCORING_COMPLETE.json').read_text())
    assert done['summary_sha256']==sha(OUT/'summary.json')
    summary=json.loads((OUT/'summary.json').read_text());rows=summary['metrics'];subsets=summary['subsets']
    def vals(group,k,key):return np.array([r[key] for r in rows if r['group']==group and r['k_source']==k],float)
    def sv(group,k,subset,key):return np.array([r[key] for r in subsets if r['group']==group and r['k_source']==k and r['subset']==subset and r[key] is not None],float)
    def fmt(a,pct=False):
        a=np.asarray(a)*(100 if pct else 1)
        return f'{a.mean():.5f} +/- {a.std(ddof=1):.5f}' if len(a)>1 else (f'{a[0]:.5f}' if len(a) else 'NA')
    lines=['Houston unsupervised transport feasibility V1: original C Joint BN+C, seeds2100/2101/2102, epoch200.',
        'One-step raw CLS transforms; classifier/head unchanged. All F0-F3 stats/routing/features/predictions frozen before GT/oracle reference.',
        'F0 identity; F1 hard means/hard routing; F2 posterior-weighted means/posterior routing;',
        'F3 same soft means, all7 hypotheses, softmax of own-class raw logits with T1, weighted candidate fusion.',
        'F3 diagonal hypothesis score equals native linear-head logit plus fixed class bias. No independent new class evidence.',
        'No iteration, temperature/threshold/fusion/k search, training/backward or later experiment.',
        'Source2393 original train, target52901; k5/15 both kept. Transport rawZ; source cosine kNN normalized separately.',
        'Oracle G2 imported after F0-F3 freeze for reference only. Oracle gap ratios un-clipped; no method selection.',
        'Means +/- sample SD of3 optimization seeds; native frozen classifier correctness defines fixed subsets.','']
    for group in GROUPS:
        lines.append(group+' classifier OA='+fmt(vals(group,5,'classifier_OA'))+'%; AA='+fmt(vals(group,5,'classifier_AA'))+'%; Kappa='+fmt(vals(group,5,'classifier_Kappa')))
        for k in KS:
            lines.append(f'  sourcek{k}: OA='+fmt(vals(group,k,'source_knn_OA'))+'%; AA='+fmt(vals(group,k,'source_knn_AA'))+'%; Kappa='+fmt(vals(group,k,'source_knn_Kappa'))+
                         '; support micro='+fmt(vals(group,k,'source_correct_support_micro'),True)+'%; macro='+fmt(vals(group,k,'source_correct_support_macro'),True)+'%')
    for subset in ['original_error','original_correct','C1_all','C7_all','C1_original_error','C7_original_error','C7_to_C6']:
        lines.extend(['',subset])
        for group in GROUPS:
            lines.append(group+' classifier accuracy='+fmt(sv(group,5,subset,'classifier_after_accuracy'),True)+'%; '
                         'wrong->correct n='+str(sv(group,5,subset,'classifier_wrong_to_correct_n').astype(int).tolist())+
                         '; correct->wrong n='+str(sv(group,5,subset,'classifier_correct_to_wrong_n').astype(int).tolist()))
            for k in KS:
                lines.append(f'  sourcek{k}: support='+fmt(sv(group,k,subset,'source_support_after'),True)+'%; '
                    'margin='+fmt(sv(group,k,subset,'source_margin_after'))+'; knn accuracy='+fmt(sv(group,k,subset,'source_knn_after_accuracy'),True)+'%; '
                    'knn wrong->correct n='+str(sv(group,k,subset,'source_knn_wrong_to_correct_n').astype(int).tolist())+
                    '; knn correct->wrong n='+str(sv(group,k,subset,'source_knn_correct_to_wrong_n').astype(int).tolist()))
    lines.extend(['','Gap capture: same-seed (F-F0)/(OracleG2-F0); no clipping.'])
    for group in GROUPS:
        for k in KS:
            for metric in ['classifier_OA','classifier_AA','source_knn_OA','source_knn_AA']:
                rr=[r['oracle_gap_fraction'] for r in summary['gap_capture'] if r['group']==group and r['k_source']==k and r['metric']==metric and r['oracle_gap_fraction'] is not None]
                lines.append(f'{group} k{k} {metric}: '+fmt(rr,True)+'% gap; seeds='+str(rr))
    lines.extend(['','Limitations:',*summary['limitations']])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    findings=['Houston unsupervised transport feasibility V1：固定F0–F3与Oracle G2参考。',
        '所有无标签target统计、routing、候选features与predictions先冻结，再导入G2并读取target GT评分。',
        'F3沿用F2 soft均值；候选自身类别raw logit评分、T1；线性head下等价于固定类别偏置重加权。',
        '没有训练、反传、调参、迭代伪标签或后续实验；两个source k完整报告。','']
    for group in GROUPS:
        findings.append(f'{group}: classifier OA={fmt(vals(group,5,"classifier_OA"))}%，AA={fmt(vals(group,5,"classifier_AA"))}%，Kappa={fmt(vals(group,5,"classifier_Kappa"))}。')
        for k in KS:
            findings.append(f'  source-kNN k{k}: OA={fmt(vals(group,k,"source_knn_OA"))}%，AA={fmt(vals(group,k,"source_knn_AA"))}%。')
        findings.append('  原classifier错误恢复='+fmt(sv(group,5,'original_error','classifier_recovery_fraction_of_subset'),True)+'%；原正确破坏='+fmt(sv(group,5,'original_correct','classifier_damage_fraction_of_subset'),True)+'%。')
    findings.extend(['','Oracle gap：按同seed比值计算均值±std，不裁剪负值。'])
    for group in ['F1','F2','F3']:
        for metric in ['classifier_OA','classifier_AA']:
            rr=[r['oracle_gap_fraction'] for r in summary['gap_capture'] if r['group']==group and r['k_source']==5 and r['metric']==metric and r['oracle_gap_fraction'] is not None]
            findings.append(f'{group} {metric}: {fmt(rr,True)}% 的G2-F0 gap。')
    findings.extend(['','重点类别和C7→C6（classifier准确率）：'])
    for subset in ['C1_all','C7_all','C7_to_C6']:
        findings.append(subset+': '+'; '.join(f'{g}={fmt(sv(g,5,subset,"classifier_after_accuracy"),True)}%' for g in GROUPS))
    findings.extend(['','无标签统计/routing的GT离线审计（不反馈构造）：'])
    for c in [1,7]:
        rr=[r for r in summary['centroid_estimation_audit'] if r['class']==c]
        findings.append(f'C{c}: hard assignment purity='+fmt([r['hard_assignment_true_class_purity'] for r in rr if r['hard_assignment_true_class_purity'] is not None],True)+
                        '%; soft purity='+fmt([r['soft_assignment_true_class_purity'] for r in rr],True)+
                        '%; hard均值L2误差='+fmt([r['hard_target_centroid_L2_error'] for r in rr])+
                        '; soft均值L2误差='+fmt([r['soft_target_centroid_L2_error'] for r in rr]))
    for group in ['F1','F2','F3']:
        rr=[r for r in summary['routing_audit'] if r['group']==group and r['subset']=='original_error']
        findings.append(f'{group}原错误样本真类routing mass='+fmt([r['true_class_routing_weight'] for r in rr],True)+
                        '%; routing argmax正确率='+fmt([r['routing_argmax_correct_fraction'] for r in rr],True)+'%。')
    findings.extend(['','解释边界：',
        'G2证明真类均值校正有oracle可修复空间，不能证明无标签目标的类别统计或routing可识别。',
        '估计类别均值和query routing相互关联；本测试没有额外的oracle-statistics/estimated-routing交叉消融，不能严格分解各自因果贡献。',
        'F3的固定偏置重加权没有新增query语义信息，不能称其已经解决多假设类别辨识。',
        '差结果只限制当前固定one-step方案，不证明所有无监督transport/routing不可行。',
        '全部结果仅诊断，不选择温度、threshold、融合系数、k、checkpoint或正式方法。'])
    (OUT/'FINDINGS.txt').write_text('\n'.join(findings)+'\n')
    aggregate=[]
    for group in GROUPS:
        for k in KS:
            for subset in sorted({r['subset'] for r in subsets if r['group']==group and r['k_source']==k}):
                rr=[r for r in subsets if r['group']==group and r['k_source']==k and r['subset']==subset]
                for key in rr[0]:
                    if key in ['seed','group','k_source','subset']:continue
                    vv=[r[key] for r in rr if r[key] is not None]
                    if vv:aggregate.append({'group':group,'k_source':k,'subset':subset,'metric':key,'available_seeds':len(vv),
                        'mean':float(np.mean(vv)),'sample_std':float(np.std(vv,ddof=1)) if len(vv)>1 else None,'values':vv})
    with (OUT/'subset_seed_aggregate.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    plot(vals,sv)
    dump(OUT/'FINAL_DELIVERY.json',{'no_training_backward_tuning_followup':True,
        'report_sha256':sha(OUT/'REPORT.txt'),'findings_sha256':sha(OUT/'FINDINGS.txt'),
        'figure_sha256':sha(OUT/'unsupervised_transport_summary.png'),'reporting_code_sha256':sha(HERE/'report.py'),
        'unsupervised_freeze_sha256':sha(OUT/'ALL_UNSUPERVISED_FROZEN.json'),'scoring_complete_sha256':sha(OUT/'SCORING_COMPLETE.json')})
    print('\n'.join(findings[:21]),flush=True)

def plot(vals,sv):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True);x=np.arange(5);width=.25
    for off,key,label,color,k in [(-width,'classifier_OA','Frozen classifier','#457b9d',5),(0,'source_knn_OA','Source kNN5','#2a9d8f',5),(width,'source_knn_OA','Source kNN15','#e9c46a',15)]:
        a=np.array([vals(g,k,key) for g in GROUPS]);axes[0].bar(x+off,a.mean(1),width,label=label,color=color,yerr=a.std(1,ddof=1),capsize=3)
    axes[0].set(title='All-target OA',ylabel='Accuracy (%)',xticks=x,xticklabels=['F0','F1','F2','F3','Oracle G2'],ylim=(0,105));axes[0].legend(fontsize=8)
    for off,subset,key,label,color in [(-.18,'original_error','classifier_recovery_fraction_of_subset','Original-error recovery','#2a9d8f'),(.18,'original_correct','classifier_damage_fraction_of_subset','Original-correct damage','#e76f51')]:
        a=np.array([sv(g,5,subset,key) for g in GROUPS])*100
        axes[1].bar(x+off,a.mean(1),.36,label=label,color=color,yerr=a.std(1,ddof=1),capsize=3)
    axes[1].set(title='Frozen classifier recovery/damage',ylabel='Subset samples (%)',xticks=x,xticklabels=['F0','F1','F2','F3','Oracle G2']);axes[1].legend(fontsize=8)
    for off,subset,label,color in [(-.18,'C1_all','C1','#457b9d'),(.18,'C7_all','C7','#e9c46a')]:
        a=np.array([sv(g,5,subset,'classifier_after_accuracy') for g in GROUPS])*100
        axes[2].bar(x+off,a.mean(1),.36,label=label,color=color,yerr=a.std(1,ddof=1),capsize=3)
    axes[2].set(title='Classifier class recall',ylabel='Recall (%)',xticks=x,xticklabels=['F0','F1','F2','F3','Oracle G2'],ylim=(0,105));axes[2].legend()
    fig.suptitle('Houston frozen raw CLS; 3 seeds mean +/- SD. One-step fixed variants; no tuning.\nOracle G2 imported only after unsupervised construction freeze.')
    fig.savefig(OUT/'unsupervised_transport_summary.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
