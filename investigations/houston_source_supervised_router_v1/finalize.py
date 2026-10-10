"""Post-scoring descriptive delivery; no training or parameter changes."""
import csv
import json
import numpy as np
from common import HERE,OUT,KS,SEEDS,sha,dump,csvwrite

s=json.loads((OUT/'summary.json').read_text())
source=list(csv.DictReader((OUT/'source_selector_diagnostic.csv').open()))

def agg(k,method):return next(r for r in s['seed_aggregate'] if r['k']==k and r['method']==method)
def sub(k,name):return next(r for r in s['subset_aggregate'] if r['k']==k and r['method']=='G3' and r['subset']==name)
def fmt(r,key):
    m=r[key+'_mean'];sd=r[key+'_sample_std']
    return 'NA' if m is None else f'{m:.3f}±{sd:.3f}' if sd is not None else f'{m:.3f}±NA'

lines=['结论：'+s['decision']+'。本轮固定source-supervised tiny selector未产生稳定target收益，按预先规则停止；不追加训练。',
 '主模型仍为原Joint BN+C/Transformer/classifier，epoch200不变。仅训练808参数MLP，hidden16、固定60 epoch。',
 '全部target输入/决策/transport及最终预测在读取target GT前冻结并hash；D00 pseudo class statistics未改变。','',
 '目标不是只判断source是否可以拟合，而是判断source学到的可靠度能否安全迁移到target。','']
for k in KS:
    lines.append(f'k={k}，3seed mean±sampleSD：')
    for method in ['G0','D00_R0','G1','G2','G3']:
        r=agg(k,method)
        lines.append(f'{method}: OA='+fmt(r,'OA')+'%; AA='+fmt(r,'AA')+'%; Kappa='+fmt(r,'Kappa')+'; 同seed ΔOA='+fmt(r,'delta_OA_vs_originalC')+'pp')
    r=agg(k,'G3')
    lines.append('G3 keep='+fmt(r,'keep_percent')+'%; override='+fmt(r,'override_percent')+'%; 实际routing类别改变='+fmt(r,'changed_routing_percent')+'%.')
    lines.append('G3原错误恢复='+fmt(r,'error_recovery_percent')+'%; 原正确破坏='+fmt(r,'correct_damage_percent')+'%. 两者分母不同，须比较样本数。')
    lines.append('平均每seed恢复='+fmt(r,'n_recovered')+'个；破坏='+fmt(r,'n_damaged')+'个；净正确数变化='+fmt(r,'net_correct_count')+'个。')
    r6=sub(k,'C6');r7=sub(k,'C7');r1=sub(k,'C1');pair=sub(k,'C7_to_C6')
    lines.append('C6平均恢复='+fmt(r6,'n_recovered')+'个，破坏='+fmt(r6,'n_damaged')+'个；C6 recall变化='+fmt(r6,'accuracy_delta_pp')+'pp。')
    lines.append('C1 recall变化='+fmt(r1,'accuracy_delta_pp')+'pp；C7 recall变化='+fmt(r7,'accuracy_delta_pp')+'pp；C7→C6原错误恢复='+fmt(pair,'error_recovery_percent')+'%.')
    lines.append('本轮并非大面积改写全部target，但少量override仍不够安全；多数损失来自C6，超过C7等类别的恢复。')
lines.extend(['','同seed G3 ΔOA（pp）：'])
for seed in SEEDS:
    lines.append(str(seed)+': '+'; '.join(f'k{k}='+str(round(next(r for r in s['target_metrics'] if r['seed']==seed and r['k']==k and r['method']=='G3')['delta_OA_vs_originalC'],5)) for k in KS))
lines.extend(['','Source holdout诊断（不参与选模或参数选择）：'])
source_rows=[]
for k in KS:
    gains=[]
    for seed in SEEDS:
        rs=[r for r in source if r['k']==str(k) and r['seed']==str(seed) and r['split']=='heldout' and r['view']!='original']
        gain=float(np.mean([float(r['selector_routing_OA'])-float(r['R0_OA']) for r in rs]));gains.append(gain)
        source_rows.append({'seed':seed,'k':k,'views':'3 shifted-source holdout views,127 origins','mean_routing_OA_gain_pp':gain})
    lines.append(f'k{k}三种shifted-source holdout视图平均routing增益={np.mean(gains):.3f}±{np.std(gains,ddof=1):.3f}pp。')
lines.extend(['source holdout有描述性routing改善，但target所有seed均损失；这说明本轮训练策略尚未实现可靠度迁移，不能断言所有source-supervised selector都不可行。',
 '固定强度虽是轻微输入扰动，affine_low使source R0 OA降至约41%–51%；因此实际表征扰动很强，其错误结构未必对应target。未据此重新调增强。',
 'G2的k5平均OA有小幅改善，但2102下降；k15平均下降，也不满足稳定收益。不能从target结果中挑k5作为正式方法。',
 'G1本轮对六个V/T/Z alternatives投票；与上轮R1–R4 majority不同，规则已在构造前固定。',
 'G2 confidence source-ECDF只统一尺度，并非GT校准概率。G2选择alternative约86%–87%，实际类别改变仅约8%–11%。',
 'oracle-selector只有GT辅助的routing可用性上界，不是transport后classifier的可达精度。',
 '只涉及一个Houston域对和3模型seed，无跨场景或显著性结论。所有方法、epoch、k和增强结果都保留。',
 '没有自动追加复杂gate、训练或任何后续实验。'])
csvwrite('source_holdout_gain_per_seed.csv',source_rows)
old=(OUT/'FINDINGS.txt').read_text();(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n\n'+old)
with (OUT/'REPORT.txt').open('a') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
gate=json.loads((OUT/'ALL_TARGET_DECISIONS_FROZEN.json').read_text())
for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
dump(OUT/'FINAL_DELIVERY.json',{'decision':s['decision'],'no_automatic_followup':True,
 'post_scoring_descriptive_text_only':True,'original_freeze_and_scoring_manifests_preserved':True,
 'finalize_code_sha256':sha(HERE/'finalize.py'),
 'files_sha256':{n:sha(OUT/n) for n in ['FINDINGS.txt','REPORT.txt','summary.json','source_supervised_router_summary.png','source_holdout_gain_per_seed.csv','ALL_TARGET_DECISIONS_FROZEN.json','SCORING_COMPLETE.json']}})
print('\n'.join(lines[:30]))
