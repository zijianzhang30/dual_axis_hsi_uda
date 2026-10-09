"""Export audited class/representation tables and standalone HTML/figure."""
import csv
import html
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/investigations/mluda_houston_m2_analysis_v1')
LOCAL=Path('/home/zhangzj26/dual_axis_hsi_uda/investigations/mluda_houston_m2_analysis_v1')
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def csv_write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

a=json.loads((OUT/'class_analysis.json').read_text());features=[json.loads((OUT/f'feature_comparison_{s}.json').read_text()) for s in [1174,1370,1417]]
rows=[];flows=[];conf=[]
for r in a['runs']:
 for c in r['classes']:
  rows.append({'seed':r['seed'],**{k:v for k,v in c.items() if k!='error_flows'}})
  flows.extend({'seed':r['seed'],**f} for f in c['error_flows'])
 for k in range(7):
  for j in range(7):
   conf.append({'seed':r['seed'],'true_class':k+1,'predicted_class':j+1,'M0_count':r['confusion_M0'][k][j],'M2_count':r['confusion_M2'][k][j],'delta_count':r['confusion_delta'][k][j]})
csv_write('per_class.csv',rows);csv_write('error_flows.csv',flows);csv_write('confusion_changes.csv',conf)
fr=[]
for f in features:
 for name,c in f['comparisons'].items():
  for k in c['classes']:
   fr.append({'seed':f['seed'],'comparison':name,'class':k['class'],'n':k['n'],
    'norm_left_mean':k['left_norm']['mean'],'norm_right_mean':k['right_norm']['mean'],'norm_left_std':k['left_norm']['std'],'norm_right_std':k['right_norm']['std'],
    'cosine_mean':k['cosine']['mean'],'cosine_std':k['cosine']['std'],'sample_relative_shift_mean':k['relative_shift']['mean'],
    **{x:v for x,v in k.items() if x.startswith('class_mean_')}})
csv_write('feature_statistics.csv',fr)
mean_delta=np.mean([[c['delta_recall_pp'] for c in r['classes']] for r in a['runs']],axis=0)
mean_oa=np.mean([[c['delta_OA_pp'] for c in r['classes']] for r in a['runs']],axis=0)
means=[{'class':k+1,'delta_recall_mean_pp':float(mean_delta[k]),'delta_AA_mean_pp':float(mean_delta[k]/7),'delta_OA_mean_pp':float(mean_oa[k])} for k in range(7)]
dump(OUT/'aggregate_class_contributions.json',means)
fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
for axrow,r in zip(axes,a['runs']):
 for ax,key,title in zip(axrow,['delta_recall_pp','delta_OA_pp'],['Recall delta (pp)','Native OA contribution (pp)']):
  v=[c[key] for c in r['classes']];ax.bar(range(1,8),v,color=['#238b45' if x>=0 else '#cb181d' for x in v]);ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(1,8));ax.set_xlabel('Class');ax.set_title(f"Seed {r['seed']}: {title}")
fig.savefig(OUT/'class_contributions.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
limit=max(abs(x) for r in a['runs'] for row in r['confusion_delta'] for x in row)
for ax,r in zip(axes,a['runs']):
 m=ax.imshow(r['confusion_delta'],cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set_title(f"Seed {r['seed']}: M2 - M0 counts");ax.set_xticks(range(7),range(1,8));ax.set_yticks(range(7),range(1,8));ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
fig.colorbar(m,ax=axes,label='Count change');fig.savefig(OUT/'confusion_changes.png',dpi=160);plt.close(fig)
content=['<h1>完整 MLUDA Houston：M2 vs M0 固定 epoch100 分析</h1>',
 '<p>仅分析已冻结的 seeds1174/1370/1417。未重训、未选 target-best、未修改 Joint BN 或损失。推理前后模型状态不变。诊断采用两模型共同的确定性FP32推理设置，与原生训练时缓存的CUDA算法不同，因此少量边界预测可能不同；原OA/recall/confusion全部仍使用原冻结预测，不用本次诊断预测替代。最终源配对通过原生 CPU sampler 随机流重放恢复，未执行训练前向或 optimizer。</p>',
 '<h2>结论与 R 启动条件</h2><p><strong>条件不满足，不启动 R。</strong>1174 的 AA 本身下降；占约61%的 C6 在1174/1417 recall 提升，仅1370下降。1174/1417主要损失来自 C7→C6 增加，1370则减少该错误。因此不存在三个seed一致的“弱类获益、大类受损”模式。</p>',
 '<p>跨seed的平均 AA 增益主要由仅22个测试像素的C4贡献：其ΔRecall均值约19.70pp，对AA贡献约+2.81pp，超过净AA提升+2.56pp。该收益并非三seed均出现（0、18.18、40.91pp）。C3在三个seed中recall均下降，C7变化方向不一致。</p>',
 '<h2>逐seed OA/AA 变化</h2>',table(['seed','ΔOA pp','ΔAA pp'],[[r['seed'],f"{r['delta_oa']:+.4f}",f"{r['delta_aa']:+.4f}"] for r in a['runs']]),
 '<h2>逐类目标占比、recall 与 OA 贡献</h2>',
 '<p>p_full是完整53200目标标签像素的占比。原生drop_last只评估53184像素；p_native=n_k/53200用于ΔOA_k=p_native×ΔRecall_k，贡献之和精确等于原生ΔOA。p_evaluated=n_k/53184及完整p均保存在CSV；p_native之和为53184/53200。</p>',
 '<img src="class_contributions.png" alt="Class contributions">']
for r in a['runs']:
 content+=['<h3>seed '+str(r['seed'])+'</h3>',table(['Class','n evaluated','p_full %','p_native %','M0 recall %','M2 recall %','ΔRecall pp','ΔOA pp','ΔAA pp'],[[c['class'],c['target_n_evaluated'],f"{100*c['target_p_full']:.4f}",f"{100*c['p_native_OA']:.4f}",f"{c['recall_M0']:.3f}",f"{c['recall_M2']:.3f}",f"{c['delta_recall_pp']:+.3f}",f"{c['delta_OA_pp']:+.4f}",f"{c['delta_AA_pp']:+.4f}"] for c in r['classes']])]
 content+=['<h4>主要增加/减少的错误流向</h4>',table(['True→predicted','M0 count','M2 count','ΔCount','Δ conditional pp'],[[f"C{f['true_class']}→C{f['predicted_class']}",f['M0_count'],f['M2_count'],f['delta_count'],f"{f['delta_conditional_percent']:+.3f}"] for f in sorted([f for c in r['classes'] for f in c['error_flows']],key=lambda f:abs(f['delta_count']),reverse=True)[:10]])]
content+=['<h2>Confusion变化</h2><img src="confusion_changes.png" alt="Confusion changes"><p>完整7×7 counts与逐类条件百分比见JSON/CSV。另保存逐像素真实类别×M0预测×M2预测的三维迁移表，可区分错误修复与新错误。</p>',
 '<h2>池化表示比较</h2><p>均在CA/SA之后、原生交叉注意力之前。cross_checkpoint比较包含CNN/注意力共同训练及通道基变化，不能当成纯池化的因果效应。within_M2在同一M2的相同CNN值上比较avg与relation，可隔离直接池化差异，但不是一个重新训练的M0反事实。类均值向量288维完整保存在JSON。</p>']
for f in features:
 content+=['<h3>seed '+str(f['seed'])+'</h3>','<p>诊断预测与冻结预测差异：'+html.escape(str({k:v['mismatches'] for k,v in f['prediction_reproduction'].items()}))+' /53184。具体索引及诊断logit边界见JSON。</p>',table(['Comparison','Left L2 norm mean','Right L2 norm mean','Sample cosine mean','Relative shift mean'],[[name,f"{c['left_norm']['mean']:.4f}",f"{c['right_norm']['mean']:.4f}",f"{c['cosine']['mean']:.6f}",f"{c['relative_shift']['mean']:.4f}"] for name,c in f['comparisons'].items()])]
 for name,c in f['comparisons'].items():
  content+=['<h4>'+name+'</h4>',table(['Class','Left norm','Right norm','Sample cosine','Mean cosine','Mean shift L2','Mean relative shift'],[[k['class'],f"{k['left_norm']['mean']:.3f}",f"{k['right_norm']['mean']:.3f}",f"{k['cosine']['mean']:.6f}",f"{k['class_mean_cosine']:.6f}",f"{k['class_mean_shift_norm']:.3f}",f"{k['class_mean_relative_shift']:.3f}"] for k in c['classes']])]
content+=['<h2>文件</h2><p>per_class.csv、error_flows.csv、confusion_changes.csv、feature_statistics.csv、class_analysis.json、feature_comparison_*.json、pool_features_*.npz。全部是固定epoch100诊断。未自动启动任何结构实验。</p>']
html_doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MLUDA Houston M2 analysis</title><style>body{max-width:1300px;margin:32px auto;font:16px/1.6 sans-serif;padding:20px;color:#17202a}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}td,th{border:1px solid #ddd;padding:6px;text-align:right}th{background:#eef3f8}img{max-width:100%}h2{margin-top:40px}</style><body>'+''.join(content)+'</body></html>'
(OUT/'report.html').write_text(html_doc)
for p in LOCAL.glob('*.py'):shutil.copy2(p,OUT/p.name)
dump(OUT/'status.json',{'state':'completed','seeds':[1174,1370,1417],'feature_data_source_pair_and_checkpoint_verified':True,'diagnostic_prediction_mismatches':{str(f['seed']):{k:v['mismatches'] for k,v in f['prediction_reproduction'].items()} for f in features},'R_started':False,'training_started':False})
print('Report complete:',OUT/'report.html')
