import json,csv,numpy as np
from scipy.special import logsumexp
import common as s

def metrics(z,y):
 z=np.asarray(z,np.float64);pred=z.argmax(1);ce=logsumexp(z,axis=1)-z[np.arange(len(y)),y];other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=z[np.arange(len(y)),y]-other.max(1);counts=np.bincount(y,minlength=7);rec=[float((pred[y==k]==k).mean()*100) if counts[k] else None for k in range(7)]
 return {'NLL':float(ce.mean()),'OA':float((pred==y).mean()*100),'AA_present_classes':float(np.mean([v for v in rec if v is not None])),'GT_logit_margin':float(margin.mean()),'class_n':counts.tolist(),'class_recall':rec,'class_NLL':[float(ce[y==k].mean()) if counts[k] else None for k in range(7)]}

def summarize(rows):
 res={'queries':len(rows),'mean_delta_vs_A':{},'NLL_sign_patterns':{},'OA_sign_patterns':{}}
 for k in ('B','C','C_normmatch'):
  res['mean_delta_vs_A'][k]={m:float(np.mean([r[k+'_minus_A'][m] for r in rows])) for m in ('NLL','OA','AA_present_classes','GT_logit_margin')}
  for metric,tol,dest in [('NLL',1e-5,'NLL_sign_patterns'),('OA',0.,'OA_sign_patterns')]:
   a=[r[k+'_minus_A'][metric] for r in rows];res[dest][k]={'beneficial':sum(v < -tol if metric=='NLL' else v>tol for v in a),'harmful':sum(v>tol if metric=='NLL' else v < -tol for v in a),'neutral':sum(abs(v)<=tol for v in a)}
 res['pred_harm_oracle_benefit_NLL']=sum(r['B_minus_A']['NLL']>1e-5 and r['C_minus_A']['NLL'] < -1e-5 for r in rows);res['pred_harm_normmatched_oracle_benefit_NLL']=sum(r['B_minus_A']['NLL']>1e-5 and r['C_normmatch_minus_A']['NLL'] < -1e-5 for r in rows);res['both_pred_and_oracle_harm_NLL']=sum(r['B_minus_A']['NLL']>1e-5 and r['C_minus_A']['NLL']>1e-5 for r in rows);return res

def main():
 s.verify();rows=[];man=[]
 for seed in s.SEEDS:
  f=s.OUT/('seed_%d/frozen_manifest.json'%seed);m=json.loads(f.read_text());assert m['original_state_and_checkpoint_hashes_unchanged'] and not m['updated_weights_saved'];assert all(s.sha(p)==h for p,h in m['prediction_files'].items());man.append(m)
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all36_local_queries_frozen':True,'all_labels_in_gradient_isolated_Oracle_updatebatch_only':True,'holdout_never_in_gradient':True,'updated_models_not_saved':True})
 y=np.load(s.OUT/'holdout_target_labels_OFFLINE_ONLY.npy');sourcey=None
 for seed in s.SEEDS:
  c=s.module();model,ls,helper,classes,audit=c.setup('houston','C',seed);sourcey=np.asarray(ls[1].dataset.labels);folder=s.OUT/('seed_%d'%seed);queries=json.load(open(folder/'gradient_queries.json'))
  for q in queries:
   values={}
   for kind in ('base','A','B','C','C_normmatch'):
    p=np.load(folder/'predictions'/('%03d_%d_%s.npz'%(q['epoch'],q['batch'],kind)));values[kind]={'target':metrics(p['target_logits'],y),'source_val':metrics(p['source_logits'],sourcey)}
   r={'seed':seed,'epoch':q['epoch'],'batch':q['batch'],'metrics':values,'gradient_diagnostic':q}
   for kind in ('B','C','C_normmatch'):
    r[kind+'_minus_A']={m:values[kind]['target'][m]-values['A']['target'][m] for m in ('NLL','OA','AA_present_classes','GT_logit_margin')}
   r['A_minus_base']={m:values['A']['target'][m]-values['base']['target'][m] for m in ('NLL','OA','AA_present_classes','GT_logit_margin')};rows.append(r)
 summary={'pooled_local_queries':summarize(rows),'per_seed':{str(seed):summarize([r for r in rows if r['seed']==seed]) for seed in s.SEEDS},'per_phase':{str(epoch):summarize([r for r in rows if r['epoch']==epoch]) for epoch in s.PHASES},'holdout_class_counts':np.bincount(y,minlength=7).tolist(),'explicit_Oracle_not_a_training_method_or_formal_performance':True,'limitations':'36 correlated finite steps on4frozen CE states,not historicaltrainingtrajectory. Oracle onehot changes confidence and activeclasses aswellas truth. Normcontrol equalizes align-gradient size not totalcombinedgradient;BNfixed postforward common. Source-targetpatchspatialcorrelation remains. No automaticlossdesign.'};s.dump(s.OUT/'summary.json',summary);s.dump(s.OUT/'scored_queries.json',rows)
 rr=[]
 for r in rows:
  for kind in ('A','B','C','C_normmatch'):
   v=r['metrics'][kind]['target'];rr.append({'seed':r['seed'],'epoch':r['epoch'],'batch':r['batch'],'variant':kind,**{k:v[k] for k in ('NLL','OA','AA_present_classes','GT_logit_margin')},'NLL_vs_A':v['NLL']-r['metrics']['A']['target']['NLL'],'OA_vs_A':v['OA']-r['metrics']['A']['target']['OA']})
 with (s.OUT/'diagnostic_deltas_NOT_FORMAL_RESULTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 s.verify();s.dump(s.OUT/'status.json',{'status':'completed_offline_counterfactual_no_followups','Oracle_results_are_not_formal_model_performance':True,'new_training_runs':0});print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
