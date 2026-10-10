"""Descriptive interpretation from scored CSVs; no GT reads or router changes."""
import csv
import json
from common import OUT,HERE,sha,dump

def read(name):return list(csv.DictReader((OUT/name).open()))
a=read('confidence_AUROC_seed_aggregate.csv');b=read('conditional_accuracy_seed_aggregate.csv')
d=read('confidence_distributions_seed_aggregate.csv')

def row(rows,k,r,sub,signal,**extra):
    return next(x for x in rows if x['k']==str(k) and x['alternative']==r and x['subset']==sub and x['signal']==signal and all(x[key]==str(val) for key,val in extra.items()))

def fmt(x,key,scale=1):
    m=x[key+'_mean'];sd=x[key+'_sample_std']
    if not m:return 'NA'
    n=x[key+'_valid_seeds']
    return f'{float(m)*scale:.3f}±{float(sd)*scale:.3f} (n={n})' if sd else f'{float(m)*scale:.3f}±NA (n={n})'

lines=['描述性结论：可用的互补routing信号，与能够识别它的confidence，是两个不同问题。',
 '当前简单自身confidence不足以支持统一覆盖R0：T kNN margin几乎不能区分纠错与损坏；V kNN有有限区分能力；R0不确定性和跨阶段agreement明显依赖类别/混淆子群。',
 '以下所有k/信号/十分位均预先固定，未从结果中选择阈值、组合权重、router或方案。','',
 '1. disagreement中“纠错 vs 损坏”的自身margin AUROC（mean±sampleSD，3 seeds）：']
for r,name,sig in [('R1','Z centroid','alternative_cosine_margin'),('R2','Z kNN','alternative_vote_margin'),
                   ('R3','T centroid','alternative_cosine_margin'),('R4','T kNN','alternative_vote_margin'),
                   ('R5','V centroid','alternative_cosine_margin'),('R6','V kNN','alternative_vote_margin')]:
    lines.append(name+': '+ '; '.join(f'k{k}='+fmt(row(a,k,r,'all_disagreement',sig),'beneficial_vs_harmful_oriented_AUROC') for k in [5,15]))
lines.extend(['','2. 预先固定的最高confidence箱（第10箱，ties未拆分；并不总是10%样本）：'])
for r,label in [('R4','T kNN'),('R6','V kNN')]:
    for k in [5,15]:
        x=row(b,k,r,'all_disagreement','alternative_vote_margin',bin=10)
        lines.append(f'{label} k{k}: alternative正确率='+fmt(x,'alternative_accuracy',100)+'%; 同一批样本R0正确率='+fmt(x,'R0_accuracy',100)+'%; 两者都错='+fmt(x,'both_wrong_fraction',100)+'%; 平均样本数='+fmt(x,'n'))
lines.extend(['T的高vote margin未体现覆盖优势。V的最高固定confidence箱有描述性的alternative优势，但不足以确立安全阈值或跨类别通用方法。','',
 '3. R0可靠度的类别依赖（R4 disagreements；方向固定为低R0 margin/高entropy更可能该覆盖）：'])
for k in [5,15]:
    for sig in ['R0_logit_margin','R0_entropy']:
        lines.append(f'k{k} {sig}: '+ '; '.join(sub+'='+fmt(row(a,k,'R4',sub,sig),'beneficial_vs_harmful_oriented_AUROC') for sub in ['all_disagreement','C1','C7']))
lines.append('总体AUROC超过0.65不代表每类有效；C7上上述预设方向AUROC低于0.4。不能依据GT子群反转方向或制定规则。')
lines.extend(['','4. C7→C6：R0必然错误，故纠错-vs-损坏AUROC为NA。alternative正确-vs-仍错的AUROC如下：'])
for k in [5,15]:
    for r,sig in [('R4','alternative_vote_margin'),('R4','alternative_agreement_other_stages'),('R6','alternative_vote_margin')]:
        lines.append(f'k{k} {r} {sig}='+fmt(row(a,k,r,'C7_to_C6',sig),'alternative_correct_raw_AUROC'))
lines.append('R4跨阶段agreement在这一GT定义的混淆子群中有描述性区分能力；该子群在部署时不能由true label直接识别，不能视为可用覆盖规则。')
lines.extend(['','5. 纠错/损坏两类confidence分布（固定R4/R6信号，median及mean）：'])
for k in [5,15]:
    for r in ['R4','R6']:
        for sig in ['alternative_vote_margin','R0_logit_margin','R0_entropy','alternative_agreement_other_stages']:
            labels=[]
            for outcome in ['beneficial_override','harmful_override']:
                x=row(d,k,r,'all_disagreement',sig,outcome=outcome)
                labels.append(outcome+' median='+fmt(x,'q50')+', mean='+fmt(x,'mean'))
            lines.append(f'k{k} {r} {sig}: '+'; '.join(labels))
lines.extend(['','冻结与实现审计：首次实现中k15的R0 margin发生临时变量复用错误，评分在输出/交付前中止；记录另存superseded_implementation_audit。',
 '最终构造修复了变量，并在最终GT评分前验证三个seed的R0 margin直接重放原logits、且k5/15的R0 margin/entropy完全相同。未使用GT改变定义、规则或参数。',
 '全部最终预测/confidence/agreements/分位边界在最终GT评分前重新冻结并hash；评分后仍保持不变。',
 '未训练、反传、调参或启动后续实验。此结果只来自一个Houston域对；三seed不是三个独立场景。'])
old=(OUT/'FINDINGS.txt').read_text();(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n\n'+old)
with (OUT/'REPORT.txt').open('a') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
audit=json.loads((OUT/'SCORING_COMPLETE.json').read_text())
assert audit['all_frozen_predictions_confidence_deciles_checkpoints_unchanged']
gate=json.loads((OUT/'ALL_ROUTER_CONFIDENCE_FROZEN.json').read_text())
for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
dump(OUT/'FINAL_DELIVERY.json',{'no_training_tuning_followup':True,
 'post_scoring_interpretation_only':True,'original_construction_and_scoring_manifests_preserved':True,
 'superseded_implementation_record':'../houston_router_reliability_diagnostic_v1_superseded_implementation_audit/SUPERSEDED.txt',
 'finalize_code_sha256':sha(HERE/'finalize.py'),
 'files_sha256':{n:sha(OUT/n) for n in ['FINDINGS.txt','REPORT.txt','router_reliability_summary.png','ALL_ROUTER_CONFIDENCE_FROZEN.json','SCORING_COMPLETE.json']}})
print('\n'.join(lines[:35]))
