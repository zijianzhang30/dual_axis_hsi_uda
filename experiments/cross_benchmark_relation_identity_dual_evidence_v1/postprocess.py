import csv,json
from collections import defaultdict
import numpy as np
import study as s

def read(name):
 with (s.OUT/name).open() as f:return list(csv.DictReader(f))
def write(name,rows):
 with (s.OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(a):return float(np.mean(a))
def sd(a):return float(np.std(a,ddof=1))
def main():
 assert (s.OUT/'GLOBAL_BEFORE_GT_FREEZE.json').exists();s.verify()
 rows=read('fixed200_per_class.csv');g=defaultdict(list)
 for r in rows:g[r['benchmark'],r['scope'],r['model'],int(r['class'])].append(r)
 out=[]
 for (d,scope,m,c),rs in sorted(g.items()):
  a=[float(r['recall']) for r in rs];out.append({'benchmark':d,'scope':scope,'model':m,'class':c,'count':int(rs[0]['count']),'recall_mean':avg(a),'recall_std':sd(a)})
 write('per_class_summary.csv',out);diff=[];zeros=[]
 for d,scope,m,c in sorted(g):
  if m!='R3':continue
  b={r['seed']:float(r['recall']) for r in g[d,scope,'R0',c]};a={r['seed']:float(r['recall']) for r in g[d,scope,'R3',c]};v=[a[seed]-b[seed] for seed in sorted(a)]
  diff.append({'benchmark':d,'scope':scope,'class':c,'delta_recall_mean':avg(v),'delta_recall_std':sd(v),'positive_seeds':sum(x>0 for x in v),'negative_seeds':sum(x<0 for x in v),'new_zero_recall_seeds':sum(a[seed]==0 and b[seed]>0 for seed in a)})
  for seed in sorted(a):
   if a[seed]==0:zeros.append({'benchmark':d,'scope':scope,'seed':seed,'class':c,'R0_recall':b[seed],'R3_recall':a[seed],'new_zero':b[seed]>0})
 write('per_class_delta_summary.csv',diff);s.dump(s.OUT/'zero_recall_audit.json',zeros)
 rates=read('recovery_damage_disagreement.csv');group=defaultdict(list)
 for r in rates:group[r['benchmark'],r['scope']].append(r)
 rr=[]
 fields=('R3_recovery_rate','R3_damage_rate','disagreement_rate','disagreement_R0_accuracy','disagreement_R1_accuracy','disagreement_R3_accuracy','both_wrong_R3_recovery_rate')
 for (d,scope),rs in sorted(group.items()):
  rr.append({'benchmark':d,'scope':scope,**{f+suffix:fn([float(r[f]) for r in rs]) for f in fields for suffix,fn in (('_mean',avg),('_std',sd))}})
 write('recovery_disagreement_summary.csv',rr)
 grid=read('same_grid_oracle_diagnostic.csv');gg=defaultdict(list)
 for r in grid:gg[r['benchmark'],r['scope'],r['seed'],r['model']].append(r)
 best=[max(rs,key=lambda r:float(r['OA'])) for key,rs in sorted(gg.items())];write('same_grid_target_best_ORACLE_ONLY.csv',best)
 decision=json.loads((s.OUT/'decision.json').read_text());summary=read('fixed200_summary.csv');delta=read('same_seed_deltas.csv')
 lines=['cross_benchmark_relation_identity_dual_evidence_v1',decision['verdict'],'PRIMARY: fixed200/native protocol, three seeds 2100/2101/2102; mean +/- sample std.','R0: immutable exact current formal C checkpoints reused. R1: independent source-CE-only spectral LN+MLP. R3: fixed normalized product, no temperature/weight/gate tuning.','Every one of 200 augmented source/target batch and original CPU/CUDA RNG streams matched baseline. HyRANK also matched per-batch indices and six RNG streams.','All six 200epoch trainings and all grid predictions verified/hash frozen before target GT scoring.','Main inference always uses epoch200. Target-best only oracle diagnostic. No subsequent experiment started.','']
 for r in summary:
  lines.append('%s / %s / %s: OA %.4f +/- %.4f; AA %.4f +/- %.4f; Kappa %.6f +/- %.6f'%(r['benchmark'],r['scope'],r['model'],*[float(r[k]) for k in ('OA_mean','OA_std','AA_mean','AA_std','Kappa_mean','Kappa_std')]))
 lines+=['','Same-seed fixed200 R3-R0:']+[json.dumps(r) for r in delta]
 lines+=['','Recovery/damage/disagreement (percent, conditional denominators documented in per-seed CSV):']+[json.dumps(r) for r in rr]
 lines+=['','Per-class R3-R0 recall changes (pp):']+[json.dumps(r) for r in diff]
 lines+=['','Zero-recall audit: '+json.dumps(zeros),'','Go checks: '+json.dumps(decision['checks']),'Systematic regression definition was locked before training: mean deltaOA<-0.5pp and >=2/3 negative OA seeds, or corresponding AA criterion.','HyRANK native10287 is main; full10317 is a separate secondary audit, never mixed.','See same_grid_target_best_ORACLE_ONLY.csv for diagnostic best epochs; no checkpoint selection in main results.','REPORT provenance: PLAN.json, code_manifest.json, baseline_manifest.json, GLOBAL_BEFORE_GT_FREEZE.json, per-run manifests/history.']
 text='\n'.join(lines)+'\n';(s.OUT/'REPORT.txt').write_text(text);(s.OUT/'FINDINGS.txt').write_text(text)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(8,3.5))
 for ax,metric in zip(axes,('OA','AA')):
  for j,d in enumerate(s.DATASETS):
   v=[float(r['delta_'+metric]) for r in delta if r['benchmark']==d and r['scope']=='native'];ax.bar(j,avg(v),yerr=sd(v),color=('#4477AA' if j==0 else '#228833'),alpha=.6,capsize=5);ax.scatter(np.linspace(j-.12,j+.12,3),v,c='black',s=24,zorder=3)
  ax.axhline(0,color='gray',lw=1);ax.set_xticks([0,1]);ax.set_xticklabels(['Pavia','HyRANK']);ax.set_ylabel('R3 - R0 '+metric+' (pp)');ax.set_title('Fixed200 '+metric)
 fig.tight_layout();fig.savefig(s.OUT/'fixed200_same_seed_deltas.png',dpi=200);fig.savefig(s.OUT/'fixed200_same_seed_deltas.pdf');plt.close(fig);print(decision['verdict'])
if __name__=='__main__':main()
