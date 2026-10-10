"""Post-scoring descriptive conclusion only; parent artifacts stay unchanged."""
import json
import csv
import numpy as np
from common import HERE,BASE,OUT,EPS,PROBES,sha,dump

s=json.loads((OUT/'summary.json').read_text())
def fmt(r,key):
    m=r[key+'_mean'];sd=r[key+'_sample_std']
    return 'NA' if m is None else f'{m:.4f}±{sd:.4f}' if sd is not None else f'{m:.4f}±NA'

lines=['结论：robustness方向 '+s['decision']+'。去除source class-discriminative component后，两个domain probe均不再稳定强于公平随机对照。',
 '原scene-shift诊断保留不变；本新增控制修正其解释：此前margin优势与source类判别成分相一致，不能单独支持特殊scene-nuisance脆弱性。',
 '三个seed的source class subspace都是rank6，projected mean/rank4 domain分别保留rank1/rank4；没有降rank或补造方向。',
 'domain与16个配对随机对照同rank、同class-orthogonal空间、同L2 norm、同source-GT最坏margin公式，epsilon固定.02/.05/.10。',
 '所有方向、扰动、logits和predictions先冻结并hash，再评分；没有读取target GT、训练、反传、调参或自动后续实验。','',
 '1. 去除前/后domain/random ratio（每seed先求比值，再3seed mean±sampleSD）：']
for probe in PROBES:
    for eps in EPS:
        r=next(r for r in s['before_after_ratio_aggregate'] if r['probe']==probe and r['epsilon']==eps and r['subset']=='all_source')
        lines.append(probe+f' epsilon={eps}: '+fmt(r,'before_domain_random_ratio')+' → '+fmt(r,'after_domain_random_ratio')+'；变化='+fmt(r,'ratio_delta'))
lines.extend(['ratio低于1意味着平均domain最坏margin drop反而小于同rank配对随机子空间最坏扰动。两个probe均未通过原mean+2 repetitionSD、3seed/全部epsilon一致性判据。',
 '本表与前轮“ratio of model-seed pooled means”的数值略有不同：这里严格使用每seed ratio后再求mean/std，原始margin-drop数据完全未改。','',
 '2. epsilon=.10投影后的margin drop（其他epsilon完整保留在REPORT/CSV）：'])
for probe in PROBES:
    r=next(r for r in s['before_after_ratio_aggregate'] if r['probe']==probe and r['epsilon']==.1 and r['subset']=='all_source')
    lines.append(probe+': domain='+fmt(r,'after_domain_margin_drop')+'；配对随机='+fmt(r,'after_random_margin_drop')+'；额外drop='+fmt(r,'after_extra_margin_drop')+'；flip增加='+fmt(r,'after_flip_increase_pp')+'pp。')
lines.extend(['','3. 残余class-between energy：'])
for probe in PROBES:
    before=next(r for r in s['energy_aggregate'] if r['probe']==probe and r['method']=='domain' and r['phase']=='before')
    after=next(r for r in s['energy_aggregate'] if r['probe']==probe and r['method']=='domain' and r['phase']=='class_orthogonal')
    lines.append(probe+': class-between fraction '+f"{before['class_between_energy_fraction_mean']:.6f} → {after['class_between_energy_fraction_mean']:.3e}; max class-basis inner-product mean={after['max_class_basis_inner_product_mean']:.3e}")
    lines.append('  centered classifier weight energy fraction '+f"{before['centered_classifier_weight_energy_fraction_mean']:.6f} → {after['centered_classifier_weight_energy_fraction_mean']:.6f}")
rank=list(csv.DictReader((OUT/'class_projection_rank_audit.csv').open()))
for probe in PROBES:
    a=np.array([float(r['retained_domain_basis_energy_fraction']) for r in rank if r['probe']==probe])
    lines.append(probe+f'投影后保留原basis Frobenius energy={100*a.mean():.3f}±{100*a.std(ddof=1):.3f}%；重新归一化/正交化后进行同范数攻击。')
lines.extend(['class-between energy接近零由该source类中心子空间的正交性保证；不等于所有非线性语义信息或target判别成分都消失。','',
 '4. 分类与各类结果：所有source train/holdout、所有7类、全部epsilon与random repetitions的OA/AA都为100%，flip rate=0%。'])
assert all(r['OA_mean']==100 and r['AA_mean']==100 and r['flip_percent_mean']==0 for r in s['metrics_aggregate'])
for probe in PROBES:
    for subset in ['C1','C7','C6']:
        r=next(r for r in s['metrics_aggregate'] if r['probe']==probe and r['epsilon']==.1 and r['method']=='CO_domain' and r['subset']==subset)
        lines.append(probe+' '+subset+' epsilon=.10 margin drop='+fmt(r,'margin_drop_mean'))
lines.extend(['','限定解释：本控制没有检出source class-orthogonal空间中的domain特异性margin优势；此前额外敏感性主要由该线性类判别成分解释的证据增强。',
 'No-Go限定于当前两种统计domain方向、原epsilon与冻结raw-Z head诊断；不证明所有scene-shift robustness方法都不可行。',
 'class basis使用全部2520 source GT，所以127 source holdout并非独立于basis构造。三个模型seed也不是三个独立场景。',
 '不自动训练robustness模型，不追加rank/epsilon/scene方向变体。'])
old=(OUT/'FINDINGS.txt').read_text();(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n\n'+old)
with (OUT/'REPORT.txt').open('a') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
gate=json.loads((OUT/'ALL_CLASS_ORTHOGONAL_PERTURBATIONS_FROZEN.json').read_text())
for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
prov=json.loads((OUT/'provenance.json').read_text());assert sha(BASE/'FINAL_DELIVERY.json')==prov['parent_final_manifest_sha256']
dump(OUT/'FINAL_DELIVERY.json',{'robustness_direction_decision':s['decision'],'target_GT_never_accessed':True,
    'no_training_backprop_tuning_followup':True,'parent_artifacts_unchanged':True,
    'post_scoring_descriptive_interpretation_only':True,'finalize_code_sha256':sha(HERE/'finalize.py'),
    'files_sha256':{n:sha(OUT/n) for n in ['FINDINGS.txt','REPORT.txt','summary.json','class_orthogonal_summary.png','class_energy_model_seed_aggregate.csv','before_after_ratios_per_seed.csv','ALL_CLASS_ORTHOGONAL_PERTURBATIONS_FROZEN.json','SCORING_COMPLETE.json']}})
print('\n'.join(lines[:30]))
