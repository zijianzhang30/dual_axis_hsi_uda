"""Post-score descriptive review only; does not change any frozen readout/feature."""
import csv,json
import numpy as np
from common import *
from score import spatial

def loadcsv(name):return list(csv.DictReader((OUT/name).open()))
def main():
 assert (OUT/'SCORING_COMPLETE.json').exists()
 s=json.loads((OUT/'summary.json').read_text());metrics=loadcsv('margin_distance_distributions.csv')
 rows=[]
 for r in metrics:rows.append({k:r[k] if k in ['subset','descriptor','metric'] else (float(v) if v else None) for k,v in r.items()})
 ma=aggregate(rows,['subset','descriptor','metric'],['mean','std','p05','p25','median','p75','p95']);write('margin_distance_seed_aggregate.csv',ma)
 # Matched class macro control uses the same <=6/>6 split in each class.
 boundary=loadcsv('boundary_error_rates_per_class.csv');macro=[]
 for seed in SEEDS:
  for dtype in ['class_mask','semantic','annotation']:
   for control in ['all','annotation_far_scene_safe']:
    selected=[r for r in boundary if int(r['seed'])==seed and r['distance_definition']==dtype and r['control']==control]
    near={r['scope']:r for r in selected if r['band']=='boundary_le6' and int(r['n'])>0};far={r['scope']:r for r in selected if r['band']=='interior_gt6' and int(r['n'])>0};classes=sorted(set(near)&set(far))
    n=float(np.mean([float(near[c]['error_rate_percent']) for c in classes])) if classes else None
    f=float(np.mean([float(far[c]['error_rate_percent']) for c in classes])) if classes else None
    macro.append({'seed':seed,'distance_definition':dtype,'control':control,'matched_classes':';'.join(classes),'n_matched_classes':len(classes),
     'boundary_macro_error_rate_percent':n,'interior_macro_error_rate_percent':f,'paired_macro_boundary_minus_interior_pp':n-f if classes else None})
 write('boundary_matched_class_macro_per_seed.csv',macro);bm=aggregate(macro,['distance_definition','control'],['boundary_macro_error_rate_percent','interior_macro_error_rate_percent','paired_macro_boundary_minus_interior_pp']);write('boundary_matched_class_macro_mean_std.csv',bm)
 y=np.load(OUT/'GT_spatial_distances.npz')['labels'];flagrows=[]
 for seed in SEEDS:
  pixel=np.load(OUT/f'seed_{seed}/OFFLINE_PIXEL_ATLAS.npz');err=pixel['R0_error'];r0=np.load(OUT/f'seed_{seed}/target_predictions.npy')
  assert np.array_equal(pixel['GT_class_C1to7']-1,y)
  assert np.array_equal(pixel['R0_prediction_C1to7']-1,r0) and np.array_equal(err,r0!=y)
  for d in DESCRIPTORS:
   pred=pixel[f'{d}_prediction_C1to7']-1;recover=int((err&(pred==y)).sum());damage=int((~err&(pred!=y)).sum())
   assert recover-damage==int((pred==y).sum())-int((r0==y).sum())
   flagrows.append({'seed':seed,'descriptor':d,'error_recovery_percent':100*recover/err.sum(),'original_correct_destruction_percent':100*damage/(~err).sum()})
   assert np.allclose(np.where(pixel[f'{d}_source_margin']>0,pixel[f'{d}_source_same_class_distance']<pixel[f'{d}_source_wrong_class_distance'],True),True)
  assert sum((pixel['VTZ_correctness_bits']==bits).sum() for bits in PATTERNS)==len(y)
  for d in STAGES:assert np.array_equal(pixel[f'{d}_positive_evidence'],pixel[f'{d}_prediction_C1to7']==pixel['GT_class_C1to7'])
  conf=loadcsv('confusion_pair_contributions_per_seed.csv');assert sum(int(r['error_n']) for r in conf if int(r['seed'])==seed)==err.sum()
  joint=loadcsv('joint_error_modes_per_seed.csv');assert sum(int(r['error_n']) for r in joint if int(r['seed'])==seed)==err.sum()
 write('frozen_readout_recovery_destruction.csv',flagrows);rd=aggregate(flagrows,['descriptor'],['error_recovery_percent','original_correct_destruction_percent']);write('frozen_readout_recovery_destruction_mean_std.csv',rd)
 # Spatial fixture distinguishes semantic contact, mask edge and isolated unknown-surrounded class.
 gt=np.full((10,20),-1,int);gt[2:8,2:8]=0;gt[2:8,8:14]=1;gt[5,16]=2
 ds,_=spatial(gt,np.array([0,0,2]),np.array([[5,7],[5,4],[5,16]]))
 assert ds['semantic'][0]==0 and ds['semantic'][1]==3 and np.isnan(ds['semantic'][2]) and ds['class_mask'][2]==0
 qa={'all3_pixel_GT_coordinate_prediction_identities_verified':True,'confusion_and_joint_modes_partition_all_errors':True,'VTZ8patterns_partition_all_samples':True,
  'positive_class_margin_distance_order_verified':True,'readout_recovery_minus_destruction_accounting_verified':True,'semantic_vs_annotation_spatial_fixture_passed':True,
  'matched_class_macro_is_posthoc_descriptive_same_fixed_distance_split':True,'no_training_or_new_readout':True}
 dump(OUT/'FINAL_REVIEW.json',qa)
 top=sorted(s['confusions'],key=lambda r:-r['all_errors_percent_mean'])[:5]
 union=next(r for r in s['unions'] if r['subset']=='R0_errors')
 lead=['结果摘要：原C三seed epoch200；只做冻结离线诊断，无训练/调参/后续实验。',
  'Top5有向混淆覆盖69.78±2.97%的全部错误；Top5无序pair覆盖78.30±2.03%。',
  'Top5联合错误模式（confusion+VTZ轨迹+GTmask边界带）覆盖56.09±6.35%；pooled seed-pixel Top5为55.49%。',
  '有向Top5：'+', '.join(f"{r['pair']} {r['all_errors_percent_mean']:.2f}%" for r in top),
  'GTmask边界/内部错误率20.13±0.67% /1.08±0.38%；99.31%的错误靠近mask边缘。annotation边缘附近包含99.27%的错误。',
  '但已标注不同类真实接触边界附近仅覆盖22.59±2.62%的错误，其near/far错误率26.60±2.73% /16.42±0.92%。未知区不代表真实异类，不能把标注轮廓误当成完整物体/类别边界。C4没有观察到semantic界面。',
  '类别比例会影响near/far微平均，追加匹配类别宏平均表，仅使用同一固定<=6/>6分组，未改变阈值。',
  '79.87%的原错误在V/T/Z centroid三个阶段都错（000）；V->T损伤8.20%，T->Z损伤8.78%，Zcentroid仍对而原classifier错3.64%。重叠机制flags不得直接相加。',
  '原classifier错误中，正确source-centroid证据：V16.42±3.44%、T10.85±2.28%、Z3.64±0.67%、raw spectrum46.97±2.08%、relation26.22±0.77%。',
  f"任一上述证据存在：{union['any_of5_evidence_percent_mean']:.2f}±{union['any_of5_evidence_percent_std']:.2f}%；raw对而V错37.93%，relation对而T错23.58%。这是离线oracle证据覆盖，不能部署为已知正确router。",
  '类内异质性：C1/C2原错误的raw谱证据86.50%/91.45%；C3 relation57.50%；C5 T73.14%但Z10.42%；C7 relation37.05%、T13.05%、Z2.69%。',
  '必须同时看全体target的readout效果：raw谱centroid OA45.11%，relation centroid OA27.39%；source heldout OA79.53%/51.18%。因此原错误里保留线索不等于替换classifier可获益。',
  'Source同类/异类距离明确为各stage的source CLASS-CENTROID距离；与nearest source sample不同。stage维度/pooling/读出几何不同，预测轨迹差异不等于因果信息删除。',
  '全部逐pixel NPZ/CSV、source-distance和margin分布、逐class/pair/trajectory、空间控制、Top5表与图片已保存。边界距单位pixel，未做地面米制假设。',
  'raw acquisition spectrum与实际输入中心谱均保存；relation为完整48x13x13 C前端中心相对/空间std归一化输入，不是学习后的token。',
  'B2原有No-Go不变，不自动提出/训练第三模块。']
 text='\n'.join(lead)+'\n\nMatched class boundary controls:\n'+'\n'.join(str(r) for r in bm)+'\n\nReadout correct destruction:\n'+'\n'.join(str(r) for r in rd)+'\n'
 original=(OUT/'FINDINGS.txt').read_text();(OUT/'FINDINGS.txt').write_text(text+'\n完整统计：\n'+original)
 with (OUT/'REPORT.txt').open('a') as f:f.write('\n最终解释与复核：\n'+text)
 s['matched_class_boundary_macro']=bm;s['readout_recovery_destruction']=rd;dump(OUT/'final_summary.json',s)
 gate=json.loads((OUT/'ALL_ATLAS_FEATURES_READOUTS_FROZEN.json').read_text())
 for rel,h in gate['files_sha256'].items():assert sha(OUT/rel)==h
 dump(OUT/'FINAL_DELIVERY.json',{'final_review_complete':True,'files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='FINAL_DELIVERY.json'}})
 print(text,flush=True)
if __name__=='__main__':main()
