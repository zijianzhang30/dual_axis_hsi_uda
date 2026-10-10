import json,csv,numpy as np
import study as s

def main():
 s.verify();records={};computes=[]
 for seed in s.SEEDS:
  f=s.B/'houston_shared_adapter_2x2_v1'/('CE0_%d'%seed)/'offline_predictions_frozen.json';records[('R0',seed)]=json.loads(f.read_text())['records'];old=json.loads((s.baseline(seed)/'frozen_manifest.json').read_text());computes.append({'arm':'R0','seed':seed,'parameters':378011,'train_seconds':old['train_seconds']})
  for arm in s.ARMS:
   folder=s.OUT/('%s_%d'%(arm,seed));f=folder/'offline_predictions_frozen.json';q=json.loads(f.read_text());assert q['target_GT_not_scored'] and q['running_statistics_unchanged'];records[(arm,seed)]=q['records'];mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==200 and mf['all200_data_RNG_hashes_match_R0'];computes.append({'arm':arm,'seed':seed,'parameters':378347,'train_seconds':mf['train_seconds'],'inference_grid_seconds':q['seconds'],'extra_projection_buffer_elements':3072 if arm=='R2' else 0})
 for rs in records.values():
  assert [r['epoch'] for r in rs]==s.PLAN['evaluation_grid'];assert all(s.sha(r['path'])==r['sha256'] for r in rs)
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all9_groups_20checkpoint_predictions_frozen':True,'no_target_GT_in_training_or_hyperparameters':True,'Best_OA_target_label_selection_diagnostic':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());coords=cache/'data/target_coords.npy';yp=cache/'data/target_labels.npy';assert s.sha(yp)==prov['data_sha256']['target_labels.npy'];assert s.sha(coords)==prov['data_sha256']['target_coords.npy'];labels=np.load(yp);assert len(labels)==52901
 helper=s.module('readout_metrics_only',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];perclass=[]
 for (arm,seed),rs in records.items():
  ev=[]
  for r in rs:
   a=np.load(r['path']);ev.append({'epoch':r['epoch'],'metrics':helper.metrics(labels,a['predictions'],a['logits'],7)})
  s.dump(s.OUT/('%s_%d_scored_grid.json'%(arm,seed)),ev)
  for ep,r in [('fixed200',ev[-1]),('target_best',max(ev,key=lambda x:x['metrics']['oa_percent']))]:
   v=r['metrics'];ends.append({'arm':arm,'seed':seed,'endpoint':ep,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa'],'zero_recall_classes':[i+1 for i,x in enumerate(v['per_class_percent']) if x==0],'collapse95':max(v['prediction_shares'])>=.95,'prediction_class_shares':v['prediction_shares']});perclass.extend([{'arm':arm,'seed':seed,'endpoint':ep,'class':i+1,'recall':x,'prediction_share':v['prediction_shares'][i]} for i,x in enumerate(v['per_class_percent'])])
 agg=[];paired=[]
 for ep in ('fixed200','target_best'):
  for arm in ('R0','R1','R2'):
   rr=[r for r in ends if r['arm']==arm and r['endpoint']==ep];agg.append({'endpoint':ep,'arm':arm,**{k:{'mean':float(np.mean([r[k] for r in rr])),'sample_std':float(np.std([r[k] for r in rr],ddof=1))} for k in ('OA','AA','Kappa')}})
  for seed in s.SEEDS:
   rr={r['arm']:r for r in ends if r['seed']==seed and r['endpoint']==ep}
   for arm,ref in [('R1','R0'),('R1','R2'),('R2','R0')]:paired.append({'seed':seed,'endpoint':ep,'candidate':arm,'baseline':ref,**{k+'_delta':rr[arm][k]-rr[ref][k] for k in ('OA','AA','Kappa')}})
 fixed=[r for r in paired if r['endpoint']=='fixed200' and r['candidate']=='R1'];gate={ref:{'mean_OA_positive':np.mean([r['OA_delta'] for r in fixed if r['baseline']==ref])>0,'mean_AA_positive':np.mean([r['AA_delta'] for r in fixed if r['baseline']==ref])>0,'seeds_both_positive':sum(r['OA_delta']>0 and r['AA_delta']>0 for r in fixed if r['baseline']==ref)} for ref in ('R0','R2')};gate={k:{a:bool(v) if isinstance(v,np.bool_) else v for a,v in x.items()} for k,x in gate.items()}
 s.dump(s.OUT/'summary.json',{'aggregate':agg,'paired':paired,'endpoints':ends,'compute':computes,'fixed200_gate_diagnostic':gate,'Go_requires_additional_perseed_class_review':True,'R2_parameterization_control_not_extra_expressivity':True,'timing_note':'shared GPU load and historical R0 timings differ; do not infer pure module timing overhead from train wallclock','Best_OA_target_oracle_only':True,'no_automatic_followups':True})
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',perclass)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 s.verify();s.dump(s.OUT/'status.json',{'status':'completed_all6_trainings_and_offline_scoring','followup_started':False,'no_new_benchmark':True});print('READOUT_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
