import sys,json,csv
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0,'/home/zhangzj26/dual_axis_hsi_uda/experiments/houston_oracle_lmmd_counterfactual_v1')
import common as s
OUT=s.B/'houston_lmmd_scale_stage_diagnostic_v1';KINDS=('A','B','C_cap03','C_cap10','D_fixed03')
def metrics(z,y):
 z=np.asarray(z,np.float64);pred=z.argmax(1);ce=logsumexp(z,axis=1)-z[np.arange(len(y)),y];other=z.copy();other[np.arange(len(y)),y]=-np.inf;mar=z[np.arange(len(y)),y]-other.max(1);counts=np.bincount(y,minlength=7)
 return {'NLL':float(ce.mean()),'OA':float((pred==y).mean()*100),'AA':float(np.mean([(pred[y==k]==k).mean()*100 for k in range(7) if counts[k]])),'margin':float(mar.mean()),'class_n':counts.tolist(),'class_recall':[float((pred[y==k]==k).mean()*100) if counts[k] else None for k in range(7)],'class_NLL':[float(ce[y==k].mean()) if counts[k] else None for k in range(7)],'class_margin':[float(mar[y==k].mean()) if counts[k] else None for k in range(7)]}
def summary(rows):
 r={'queries':len(rows),'arms':{},'gradient_cosine_mean':float(np.mean([q['gradient_cosine'] for q in rows])),'gradient_cosine_negative':sum(q['gradient_cosine']<0 for q in rows)}
 for k in KINDS:
  ds=[q['deltas_vs_A'][k]['target'] for q in rows];r['arms'][k]={'target_delta_vs_A_mean':{m:float(np.mean([d[m] for d in ds])) for m in ('NLL','OA','AA','margin')},'source_delta_vs_A_mean':{m:float(np.mean([q['deltas_vs_A'][k]['source_val'][m] for q in rows])) for m in ('NLL','OA','AA','margin')},'target_NLL_harmful':sum(d['NLL']>1e-5 for d in ds),'target_NLL_beneficial':sum(d['NLL']< -1e-5 for d in ds),'target_OA_harmful':sum(d['OA']<0 for d in ds),'target_OA_beneficial':sum(d['OA']>0 for d in ds),'align_CE_ratio_median':float(np.median([q['arms'][k]['weighted_LMMD_over_CE'] for q in rows])),'alignment_retained_median':float(np.median([q['arms'][k]['native_alignment_retained_fraction'] for q in rows])),'actual_update_norm_mean':float(np.mean([q['arms'][k]['actual_parameter_update_norm'] for q in rows])),'CE_gradient_norm_mean':float(np.mean([q['CE_gradient_norm'] for q in rows])),'LMMD_gradient_norm_mean':float(np.mean([q['LMMD_gradient_norm'] for q in rows]))}
 return r
def main():
 s.verify();man=[]
 for seed in s.SEEDS:
  m=json.loads((OUT/('seed_%d/frozen_manifest.json'%seed)).read_text());assert m['queries']==9 and m['checkpoint_unchanged'] and not m['target_GT_accessed_for_gradient'];assert all(s.sha(p)==h for p,h in m['files'].items());man.append(m)
 s.dump(OUT/'OFFLINE_SCORING_BARRIER.json',{'all27_queries_frozen_before_target_GT_scoring':True,'target_GT_never_in_gradients':True,'fixed_protocol_no_target_tuning':True})
 y=np.load(s.OUT/'holdout_target_labels_OFFLINE_ONLY.npy');rows=[]
 for seed,m in zip(s.SEEDS,man):
  ys=np.asarray(m['source_val_labels'])
  for q in json.loads((OUT/('seed_%d/queries.json'%seed)).read_text()):
   vals={}
   for k,v in q['predictions'].items():
    p=np.load(v['path']);vals[k]={'target':metrics(p['target_logits'],y),'source_val':metrics(p['source_logits'],ys)}
   q['metrics']=vals;q['deltas_vs_A']={k:{domain:{key:vals[k][domain][key]-vals['A'][domain][key] for key in ('NLL','OA','AA','margin')} for domain in ('target','source_val')} for k in KINDS};q['deltas_vs_base']={k:{domain:{key:vals[k][domain][key]-vals['base'][domain][key] for key in ('NLL','OA','AA','margin')} for domain in ('target','source_val')} for k in KINDS};rows.append(q)
 result={'pooled':summary(rows),'per_phase':{str(e):summary([q for q in rows if q['epoch']==e]) for e in (10,100,200)},'per_seed':{str(seed):summary([q for q in rows if q['seed']==seed]) for seed in s.SEEDS},'per_seed_phase':{str(seed):{str(e):summary([q for q in rows if q['epoch']==e and q['seed']==seed]) for e in (10,100,200)} for seed in s.SEEDS},'target_class_counts':np.bincount(y,minlength=7).tolist(),'limitations':'27 correlated single steps,8192 fixed target pixels; not long-term training or formal model accuracy. C4 only4 pixels. CE checkpoints carry no momentum (native SGD momentum0); all arms share post-single-forward BN buffers and same source/target batches,RNG; no BN updates during evaluation. Fixed coefficient changes both stage effects and relative gradient norms, so cannot uniquely identify stage causality. Caps0.3/1.0 predetermined, no target tuning. Class-margin labels used offline only. No new training or followup.'}
 s.dump(OUT/'summary.json',result);s.dump(OUT/'scored_queries.json',rows)
 csvrows=[];classrows=[]
 for q in rows:
  for k in KINDS:
   csvrows.append({'seed':q['seed'],'epoch':q['epoch'],'batch':q['batch'],'arm':k,'CE_loss':q['CE'],'LMMD_loss':q['LMMD'],'gradient_cosine':q['gradient_cosine'],**q['arms'][k],**{'target_'+key+'_vs_A':v for key,v in q['deltas_vs_A'][k]['target'].items()},**{'source_'+key+'_vs_A':v for key,v in q['deltas_vs_A'][k]['source_val'].items()}})
   for domain in ('source_val','target'):
    for c in range(7):
     a=q['metrics']['A'][domain];b=q['metrics'][k][domain];classrows.append({'seed':q['seed'],'epoch':q['epoch'],'batch':q['batch'],'arm':k,'domain':domain,'class':c+1,'n':a['class_n'][c],**{key+'_vs_A':None if a[key][c] is None else b[key][c]-a[key][c] for key in ('class_recall','class_NLL','class_margin')}})
 for name,rs in [('one_step_deltas_NOT_FORMAL_RESULTS.csv',csvrows),('per_class_margin_deltas.csv',classrows)]:
  with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 s.verify();s.dump(OUT/'status.json',{'status':'completed_offline_scale_stage_diagnostic','new_training_runs':0,'followup_started':False});print(json.dumps(result['per_phase'],indent=2),flush=True)
if __name__=='__main__':main()
