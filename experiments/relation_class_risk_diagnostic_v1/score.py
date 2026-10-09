import json,numpy as np,torch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from run import c,OUT,SEEDS,dump,metrics,sha

def summary(rows):
 harmful=np.array([r['target_micro_CE_delta']>1e-5 for r in rows]);res={'queries':len(rows),'harmful_CE_updates':int(harmful.sum()),'target_OA_decline_updates':sum(r['target_OA_delta']<0 for r in rows),'target_mean_CE_delta':float(np.mean([r['target_micro_CE_delta'] for r in rows])),'scores':{}}
 for key in ('clean_primary_score','perturbed_primary_score','alignment_update_norm'):
  s=np.array([r[key] for r in rows]);rho=spearmanr(s,[r['target_micro_CE_delta'] for r in rows]);warning=s>0
  res['scores'][key]={'AUROC_harmful_CE':float(roc_auc_score(harmful,s)) if len(set(harmful))==2 else None,'spearman_target_CE_damage':float(rho.statistic if hasattr(rho,'statistic') else rho.correlation),'positive_warning_fraction':float(warning.mean()),'warning_harmful_precision':float(harmful[warning].mean()) if warning.any() else None,'warning_false_positive_rate':float(warning[~harmful].mean()) if (~harmful).any() else None}
 return res

def main():
 barrier=json.loads((OUT/'GT_SCORING_BARRIER.json').read_text());assert barrier['all_predictions_frozen'];assert all(sha(p)==h for p,h in barrier['prediction_sha256'].items());rows=json.loads((OUT/'queries.json').read_text());allrows=[]
 for seed in SEEDS:
  m,ls,helper,classes,audit=c.setup('houston','C',seed);ids=np.load(OUT/('target_ids_%d.npy'%seed));y=torch.as_tensor(np.asarray(ls[-1].dataset.labels)[ids],dtype=torch.long)
  counts=np.bincount(y.numpy(),minlength=7);assert (counts>0).all()
  for row in [r for r in rows if r['seed']==seed]:
   states={}
   for kind in ('CE','candidate'):
    z=torch.from_numpy(np.load(OUT/'predictions'/('%d_%03d_%d_%s.npz'%(seed,row['epoch'],row['batch'],kind)))['logits']);mt=metrics(z,y);mt['micro_CE']=float(torch.nn.functional.cross_entropy(z,y));states[kind]=mt
   row.update({'target_metrics':states,'target_micro_CE_delta':states['candidate']['micro_CE']-states['CE']['micro_CE'],'target_OA_delta':states['candidate']['OA']-states['CE']['OA'],'target_macro_CE_delta':float(np.mean(states['candidate']['class_CE'])-np.mean(states['CE']['class_CE'])),'target_class_counts':counts.tolist()});allrows.append(row)
  del m,ls
 dump(OUT/'offline_scored_queries.json',allrows);ss={'pooled':summary(allrows),'per_seed':{str(seed):summary([r for r in allrows if r['seed']==seed]) for seed in SEEDS},'per_phase':{str(e):summary([r for r in allrows if r['epoch']==e]) for e in (1,10,50,100,200)},'interpretation':'Only local hypothetical updates on frozen end-of-epoch states; target GT scoring offline. No threshold adjustment; tiny steps/OA ties and within-checkpoint correlated queries limit inference. No trained constrained method or target safety guarantee.','no_auto_training':True};dump(OUT/'summary.json',ss);dump(OUT/'status.json',{'status':'completed_read_only_diagnostic_no_followups','queries':len(allrows)});print(json.dumps(ss,indent=2),flush=True)
if __name__=='__main__':main()
