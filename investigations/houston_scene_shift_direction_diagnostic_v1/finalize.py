"""Interpret source scores only; no target GT, changed thresholds or further experiments."""
import json
import csv
import numpy as np
from common import HERE,OUT,SEEDS,EPS,PROBES,sha,dump

s=json.loads((OUT/'summary.json').read_text())
def row(probe,eps,method,subset='all_source'):
    return next(r for r in s['metrics_aggregate'] if r['probe']==probe and r['epsilon']==eps and r['method']==method and r['subset']==subset)
def fmt(r,key):
    m=r[key+'_mean'];sd=r[key+'_sample_std']
    return 'NA' if m is None else f'{m:.4f}±{sd:.4f}' if sd is not None else f'{m:.4f}±NA'

lines=['结论：有稳定的directional margin敏感性证据；没有在本轮固定epsilon下观察到source分类破坏。',
 '预设margin诊断判定为'+s['decision']+'，仅是机制诊断层面的信号，不是训练方法Go或target性能提升结论。',
 '两个probe都在3个模型seed、全部epsilon上超过同rank随机子空间最坏扰动对照的mean+2重复SD；这是描述性一致性标准，不是统计显著性检验。',
 '全部source/target无标签方向先冻结，再用source GT作闭式最坏方向选择；所有扰动和预测冻结后才评分。target GT从未读取。',
 '没有训练、反传、调参或启动任何后续实验。','',
 '1. 公平比较：domain和随机子空间均使用source GT、同rank和同范数、同一最坏margin公式。',
 '普通random/orthogonal只是补充控制，不能独自排除adversarial-selection的优势。',
 '以下为3seed mean±sampleSD，margin单位是原classifier logit：']
for probe in PROBES:
    for eps in EPS:
        d=row(probe,eps,'D_domain');c=row(probe,eps,'D_random_subspace_adversarial')
        lines.append(probe+f' epsilon={eps}: domain margin drop='+fmt(d,'margin_drop_mean')+'; 同rank随机最坏='+fmt(c,'margin_drop_mean')+'; 比值(均值比)='+str(round(d['margin_drop_mean_mean']/c['margin_drop_mean_mean'],3)))
lines.extend(['','2. 分类结果：',
 '所有probe、epsilon、control、16随机重复、3模型seed：source OA/AA都为100%，prediction flip=0%。source train2393和heldout127同样未发生翻转。',
 '原source mean true-vs-strongest margin约8.9249；epsilon=.10后，mean-direction最坏约8.4442，rank4最坏约7.8374。下降可辨别，但仍未越过分类边界。',
 '所以不能写成“domain扰动已更容易造成source误分类”或“scene-shift-aware training已经有效”。本轮没有classification-flip evidence。','',
 '3. 在epsilon=.10，普通随机/正交及无label选择的domain内随机控制：'])
for probe in PROBES:
    for method in ['D_random','D_orthogonal','D_domain_random']:
        r=row(probe,.1,method);lines.append(probe+' '+method+' margin drop='+fmt(r,'margin_drop_mean')+'; flip='+fmt(r,'flip_percent')+'%.')
lines.extend(['','4. mean direction的“真实正向”与source-label最坏符号是两件事：'])
for eps in EPS:
    r=row('global_mean',eps,'D_mean_forward');d=row('global_mean',eps,'D_domain')
    lines.append(f'epsilon{eps}: 固定target-minus-source正向mean drop='+fmt(r,'margin_drop_mean')+'；逐sample选择最坏正负符号的drop='+fmt(d,'margin_drop_mean')+'。')
lines.append('最坏符号可能与真实source→target mean方向相反；后者不能被最坏符号结果替代。')
lines.extend(['','5. C1/C7/C6（epsilon=.10，完整其他epsilon在REPORT/CSV）：'])
for probe in PROBES:
    for cls in ['C1','C7','C6']:
        r=row(probe,.1,'D_domain',cls)
        lines.append(probe+' '+cls+': margin drop='+fmt(r,'margin_drop_mean')+'; OA='+fmt(r,'OA')+'%; flip='+fmt(r,'flip_percent')+'%.')
lines.extend(['','6. 语义混杂：'])
for r in s['source_semantic_audit']:
    lines.append(r['probe']+' '+r['basis_type']+': source类间能量占比='+str(round(100*r['source_between_class_energy_fraction_mean'],3))+'%; 类内='+str(round(100*r['source_within_class_energy_fraction_mean'],3))+'%; centered-classifier权重能量='+str(round(100*r['centered_classifier_weight_energy_fraction_mean'],3))+'%。')
lines.extend(['rank4 domain subspace捕获约67% source类间能量、51% centered classifier权重能量，远高于随机rank4约6%。方向敏感性可能部分来自语义判别轴与类别比例变化，未被识别为纯scene nuisance。',
 '实际两域mean shift L2约4.21–4.43；固定扰动半径只是原rawZ范数的2%/5%/10%，并非完整domain transformation。没有追加更大epsilon。',
 '此结果支持“真实分布差异轴上margin更敏感”的限定描述；不证明对应输入空间scene变化可实现、target误差原因或训练抑制这些轴的净收益。',
 '当前没有充分的分类破坏证据作为自动启动robustness training的依据；按用户要求，本轮止于诊断。'])
old=(OUT/'FINDINGS.txt').read_text();(OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n\n'+old)
with (OUT/'REPORT.txt').open('a') as f:f.write('\n\n'+'\n'.join(lines)+'\n')
gate=json.loads((OUT/'ALL_SOURCE_PERTURBATIONS_FROZEN.json').read_text())
for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
dump(OUT/'FINAL_DELIVERY.json',{'margin_diagnostic_decision':s['decision'],
  'classification_damage_observed':False,'robustness_training_benefit_established':False,
  'target_GT_never_accessed':True,'no_training_backprop_tuning_followup':True,
  'post_scoring_interpretation_only':True,'original_freeze_and_scoring_manifests_preserved':True,
  'finalize_code_sha256':sha(HERE/'finalize.py'),
  'files_sha256':{n:sha(OUT/n) for n in ['FINDINGS.txt','REPORT.txt','summary.json','scene_shift_direction_summary.png','ALL_DOMAIN_DIRECTIONS_FROZEN.json','ALL_SOURCE_PERTURBATIONS_FROZEN.json','SCORING_COMPLETE.json']}})
print('\n'.join(lines[:26]))
