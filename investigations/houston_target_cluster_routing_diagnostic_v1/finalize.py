"""Descriptive interpretation after scoring; no new routing variant or parameter choice."""
import json
import numpy as np
from common import HERE,OUT,SEEDS,STAGES,sha,dump,csvwrite

s=json.loads((OUT/'summary.json').read_text())
def fmt(r,key):
    m=r[key+'_mean'];sd=r[key+'_sample_std']
    return 'NA' if m is None else f'{m:.3f}±{sd:.3f}' if sd is not None else f'{m:.3f}±NA'
def mr(stage):return next(r for r in s['metrics_aggregate'] if r['method']==stage+'_cluster')

lines=['结论：'+s['decision']+'。当前固定K7 KMeans + Hungarian一一matching + D00-statistics transport均未带来三seed稳定正收益。',
 '没有训练、反传、调参；没有自动追加GMM、spectral、不同K或其他变体。',
 '所有features、cluster/centroid/cost/matching、routing及transport最终预测先冻结并hash，才读取target GT；oracle参考仅在评分阶段读取。','',
 '1. target cluster structure能提供局部互补，但没有形成更可靠的全局class routing：']
for stage in STAGES:
    r=mr(stage);p=next(x for x in s['purity_aggregate'] if x['stage']==stage)
    lines.append(stage+' weighted purity='+fmt(p,'weighted_purity_percent')+'%; macro cluster purity='+fmt(p,'macro_purity_percent')+'%; routing OA='+fmt(r,'routing_OA')+'%; routing AA='+fmt(r,'routing_AA')+'%; 原R0错误上的routing正确率='+fmt(r,'routing_correct_on_R0_error_percent')+'%.')
lines.append('R0 routing OA82.034%；历史T/Z source-kNN约77.7%/81.6%；source-supervised G3 routing约79.8%–80.0%。本轮cluster routing总体均更差。')
lines.extend(['','2. 固定D00 pseudo means不变，只替换routing的classifier结果：'])
for stage in STAGES:
    r=mr(stage)
    lines.append(stage+': OA='+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; Kappa='+fmt(r,'Kappa')+'; 同seed ΔOA vsD00='+fmt(r,'OA_delta_vs_D00')+'pp。')
    lines.append('原错误恢复='+fmt(r,'original_error_recovery_percent')+'%; 原正确破坏='+fmt(r,'original_correct_damage_percent')+'%; 平均恢复/破坏样本数='+fmt(r,'n_recovered')+'/'+fmt(r,'n_damaged')+'。')
    lines.append('D01 / D11 oracle OA gap closure='+fmt(r,'OA_gap_closure_vs_D01_percent')+'% / '+fmt(r,'OA_gap_closure_vs_D11_percent')+'%。负值表示gap扩大，未截断。')
lines.extend(['','3. 同seed ΔOA vsD00（pp）：'])
for seed in SEEDS:
    lines.append(str(seed)+': '+'; '.join(stage+'='+str(round(next(r for r in s['metrics'] if r['seed']==seed and r['method']==stage+'_cluster')['OA_delta_vs_D00'],5)) for stage in STAGES))
lines.extend(['','4. purity不能替代class correspondence：',
 'T每个seed的7个cluster仅覆盖5个GT majority classes，其中3个cluster均以C6为主；Z的C6-majority cluster为3/3/2个。',
 'Hungarian强制一cluster对应一source类，因此不能把这些C6子群都匹配到C6；另一些被分配到C1/C4等类。该判断是冻结后GT解释，并未用于构造或修正matching。',
 '具体例子：seed2100 T cluster5有12409个样本、98.17%属于C6，却被匹配到source C4；seed2102 Z cluster5有16021个样本、96.54%属于C6，也被匹配到C4。',
 '这表明本轮“7个几何cluster = 7个语义类”的一一对应假设不成立；不能据此断言所有无监督target结构都无价值。','',
 '5. C1/C7与主要混淆：'])
for stage in STAGES:
    for sub in ['C1_all','C7_all','C7_to_C6']:
        r=next(x for x in s['subsets_aggregate'] if x['method']==stage+'_cluster' and x['subset']==sub)
        lines.append(stage+' '+sub+': routing正确率='+fmt(r,'routing_accuracy_percent')+'%; transport classifier正确率='+fmt(r,'classifier_accuracy_percent')+'%; 原错误恢复='+fmt(r,'error_recovery_percent')+'%.')
lines.extend(['T能为部分C7→C6错误提供正确信号，但其他样本routing受损，不能形成整体安全收益。C1结果跨seed不稳定；两stage的T/Z输入和所有结果均完整保留。',
 'OA、AA、Kappa、routing与transport正确率是不同指标；T的19.1%原错误routing覆盖不表示classifier能恢复同样比例。',
 '仅一个Houston域对和三个优化seed；No-Go限定于本轮固定方案。未以target GT选择stage、K、初始化次数或任何参数。'])

# Read-only audit of the already fixed matching, not a competing router.
audit=[]
for seed in SEEDS:
    for stage in STAGES:
        a=np.load(OUT/f'seed_{seed}/{stage}_clustering_matching.npz');cost=a['cosine_cost'];mapping=a['cluster_to_source_class']
        for j in range(7):
            best=int(cost[j].argmin());assigned=int(mapping[j])
            audit.append({'seed':seed,'stage':stage,'cluster':j,'assigned_source_class':assigned+1,
                'individually_nearest_source_class':best+1,'assigned_cosine':float(1-cost[j,assigned]),
                'individually_best_cosine':float(1-cost[j,best]),'one_to_one_assignment_cost_penalty':float(cost[j,assigned]-cost[j,best])})
csvwrite('matching_constraint_audit.csv',audit)
old=(OUT/'FINDINGS.txt').read_text();(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n\n'+old)
with (OUT/'REPORT.txt').open('a') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
gate=json.loads((OUT/'ALL_CLUSTER_ROUTING_TRANSPORT_FROZEN.json').read_text())
for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
dump(OUT/'FINAL_DELIVERY.json',{'decision':s['decision'],'no_training_tuning_followup':True,
  'post_scoring_descriptive_text_and_fixed_matching_audit_only':True,'original_freeze_and_scoring_manifests_preserved':True,
  'finalize_code_sha256':sha(HERE/'finalize.py'),
  'files_sha256':{n:sha(OUT/n) for n in ['FINDINGS.txt','REPORT.txt','summary.json','matching_constraint_audit.csv','cluster_class_confusions.png','cluster_routing_transport_summary.png','ALL_CLUSTER_ROUTING_TRANSPORT_FROZEN.json','SCORING_COMPLETE.json']}})
print('\n'.join(lines[:30]))
