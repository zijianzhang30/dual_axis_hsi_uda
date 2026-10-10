import csv,json
import numpy as np
from sklearn.metrics import confusion_matrix,cohen_kappa_score
import study as s

def write(name,rows):
 with (s.OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def aggregate(rows,keys,metrics):
 groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 out=[]
 for kk,rr in groups.items():
  d=dict(zip(keys,kk))
  for m in metrics:
   values=np.array([r[m] for r in rr],float);d[m+'_mean']=float(values.mean());d[m+'_std']=float(values.std(ddof=1))
  out.append(d)
 return out
def main():
 s.verify();frozen=json.loads((s.OUT/'ALL_NINE_FINAL_PREDICTIONS_FROZEN.json').read_text());assert not frozen['target_GT_scored']
 for path,h in frozen['files_sha256'].items():assert s.sha(path)==h
 y=np.load(s.DATA/'target_labels.npy');gt=np.load(s.DATA/'target_gt.npy',mmap_mode='r');coords=np.load(s.GEOM/'seed_2100/target_coords.npy');assert np.array_equal(y,gt[coords[:,0],coords[:,1]])
 metadata=json.loads((s.GEOM/'provenance.json').read_text())['historical_metadata'];import hashlib
 assert hashlib.sha256(y.tobytes()).hexdigest()==metadata['domains']['target']['labels_sha256']
 rows=[];recalls=[];confusions=[];marginrows=[];sourcerows=[];replays=[]
 for seed in s.SEEDS:
  base=np.load(s.OUT/f'B0_{seed}'/'predictions_200.npz')['predictions'];berr=base!=y
  metrics={}
  for arm in s.ARMS:
   folder=s.OUT/f'{arm}_{seed}';pred=np.load(folder/'predictions_200.npz')['predictions'];cm=confusion_matrix(y,pred,labels=np.arange(7));rec=100*np.diag(cm)/cm.sum(1)
   r={'seed':seed,'arm':arm,'OA':100*float((pred==y).mean()),'AA':float(rec.mean()),'Kappa':float(cohen_kappa_score(y,pred,labels=np.arange(7))),
    'original_error_recovery_percent':100*float((pred[berr]==y[berr]).mean()),'original_correct_destruction_percent':100*float((pred[~berr]!=y[~berr]).mean())}
   metrics[arm]=(r,rec);rows.append(r)
   for c in range(7):recalls.append({'seed':seed,'arm':arm,'class':f'C{c+1}','recall':float(rec[c]),'n':int(cm[c].sum())})
   for a in range(7):
    for b in range(7):confusions.append({'seed':seed,'arm':arm,'true_class':f'C{a+1}','prediction':f'C{b+1}','n':int(cm[a,b]),'fraction_of_true_class_percent':100*cm[a,b]/cm[a].sum()})
   for fname,targetlist in [('source_margin_summary.csv',marginrows),('source_pair_margin_summary.csv',sourcerows)]:
    for rr in csv.DictReader((folder/fname).open()):
     converted={k:(v if k in ['split','class','pair','true_class','competitor'] else float(v)) for k,v in rr.items()};targetlist.append({'seed':seed,'arm':arm,**converted})
   mf=json.loads((folder/'FINAL_PREDICTIONS_FROZEN.json').read_text());replays.append({'seed':seed,'arm':arm,'train_seconds':mf['train_seconds'],'archived_B0_replay':mf['archived_B0_replay']})
  for arm in s.ARMS:
   r,rec=metrics[arm]
   for metric in ['OA','AA','Kappa']:r['delta_'+metric]=r[metric]-metrics['B0'][0][metric]
   r['worst_class_recall_delta']=float((rec-metrics['B0'][1]).min())
   for row in recalls:
    if row['seed']==seed and row['arm']==arm:row['delta_recall']=row['recall']-metrics['B0'][1][int(row['class'][1:])-1]
 write('fixed200_metrics_per_seed.csv',rows);write('recall_per_seed_per_class.csv',recalls);write('confusion_per_seed.csv',confusions)
 write('source_margin_per_seed_per_class.csv',marginrows);write('source_pair_margin_per_seed.csv',sourcerows)
 summary=aggregate(rows,['arm'],['OA','AA','Kappa','delta_OA','delta_AA','delta_Kappa','original_error_recovery_percent','original_correct_destruction_percent','worst_class_recall_delta'])
 ra=aggregate(recalls,['arm','class'],['recall','delta_recall']);write('fixed200_metrics_mean_std.csv',summary);write('recall_mean_std.csv',ra)
 ma=aggregate(marginrows,['arm','split','class'],[k for k in marginrows[0] if k not in ['seed','arm','split','class']]);write('source_margin_mean_std.csv',ma)
 pa=aggregate(sourcerows,['arm','pair','true_class','competitor'],[k for k in sourcerows[0] if k not in ['seed','arm','pair','true_class','competitor']]);write('source_pair_margin_mean_std.csv',pa)
 b2=[r for r in rows if r['arm']=='B2'];b2summary=next(r for r in summary if r['arm']=='B2')
 conditions={'all3_seeds_OA_positive':all(r['delta_OA']>0 for r in b2),'mean_AA_positive':b2summary['delta_AA_mean']>0,'mean_Kappa_positive':b2summary['delta_Kappa_mean']>0,
  'no_any_class_seed_recall_drop_ge5pp':all(r['delta_recall']>-5 for r in recalls if r['arm']=='B2')}
 decision='Go-for-this-fixed-minimal-validation' if all(conditions.values()) else 'No-Go'
 output={'decision':decision,'prespecified_conditions':conditions,'metrics':summary,'per_seed':rows,'recalls':ra,'baseline_replay':replays,'no_automatic_followup':True}
 s.dump(s.OUT/'summary.json',output)
 lines=['Houston boundary robustification v1; fixed200; original C Full Joint BN+tokenizer;3seeds.',
  'Nine final predictions and weights hash-frozen before new target scoring; no target GT in training/weight construction/selection.',
  'Fresh B0 paired baseline. B1 frequency; B2 activity*stress. All21pair normalized weights, boundary CE lambda1,50%signed-distance step.',
  'Unchanged source CE/SGD/data/BN/Transformer protocol; perturbation offset detached, no extra network forward.',
  'Mean+/-sampleSD across3seeds; deltas use same-seed fresh B0.','',f'DECISION: {decision}',str(conditions),'\nMetrics:']
 lines.extend(str(r) for r in summary);lines.append('\nSame-seed metrics:');lines.extend(str(r) for r in rows)
 lines.append('\nFocus class recalls:');lines.extend(str(r) for r in ra if r['class'] in ['C4','C5','C6','C7'])
 lines.append('\nSource all2520 original/perturbed margins:');lines.extend(str(r) for r in ma if r['split']=='all' and r['class']=='macro_samples')
 lines.append('\nArchived baseline replay:');lines.extend(str(r) for r in replays if r['arm']=='B0')
 lines.extend(['\nLimits:','Loss weights/stats include native existing labeled target candidate mask but never target class values. Stress uses all2520source GT including127val, matching fixed diagnostic.',
  'Pair-hyperplane proximity and absolute stress do not determine correct boundary motion;50%keeps each selected pair side but can cross a third-class boundary.',
  'Classifier and representation are trained; compare geometric and logit margins to avoid interpreting pure logit scaling as robustness.',
  '5pp per-class damage flag was fixed before training; three-seed stability is descriptive, not a significance claim.',
  'Decision applies to this fixed configuration, no formula/weight/perturbation tuning or automatic new experiments.'])
 text='\n'.join(lines)+'\n';(s.OUT/'REPORT.txt').write_text(text);(s.OUT/'FINDINGS.txt').write_text(text)
 plot(summary,ra,ma)
 for path,h in frozen['files_sha256'].items():assert s.sha(path)==h
 s.verify();s.dump(s.OUT/'SCORING_COMPLETE.json',{'all_frozen_predictions_code_weights_checkpoints_unchanged':True,'target_GT_scored_only_after_all9_frozen':True,'decision':decision,'no_auto_training':True})
 s.dump(s.OUT/'FINAL_DELIVERY.json',{'files_sha256':{str(p):s.sha(p) for p in s.OUT.rglob('*') if p.is_file() and p.name!='FINAL_DELIVERY.json' and 'logs' not in p.parts}})
 print(text,flush=True)

def plot(summary,recalls,margins):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 for i,r in enumerate(summary):axes[0].bar(i,r['OA_mean'],yerr=r['OA_std'],capsize=3)
 axes[0].set(xticks=range(3),xticklabels=s.ARMS,ylabel='Target OA (%)',title='Fixed200,3seed mean +/- SD')
 x=np.arange(7)
 for i,arm in enumerate(s.ARMS):
  rr=[r for r in recalls if r['arm']==arm];axes[1].bar(x+(i-1)*.25,[r['delta_recall_mean'] for r in rr],.25,label=arm)
 axes[1].set(xticks=x,xticklabels=[f'C{c+1}' for c in x],ylabel='Recall change vs B0 (pp)');axes[1].legend()
 for i,arm in enumerate(s.ARMS):
  r=next(r for r in margins if r['arm']==arm and r['split']=='all' and r['class']=='macro_samples')
  axes[2].bar(i-.15,r['original_signed_pair_geometric_margin_mean_mean'],.3,label='Original' if i==0 else None)
  axes[2].bar(i+.15,r['perturbed_signed_pair_geometric_margin_mean_mean'],.3,label='50% perturbed' if i==0 else None)
 axes[2].set(xticks=range(3),xticklabels=s.ARMS,ylabel='Source signed geometric pair margin');axes[2].legend()
 fig.savefig(s.OUT/'boundary_robustification_summary.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
