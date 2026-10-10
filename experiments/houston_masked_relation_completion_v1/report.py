import json,csv,numpy as np
import study as s

def main():
 s.verify();rs={'E0':[r for r in json.loads((s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json').read_text())['records'] if r['epoch']<=100]};computes={};hh={}
 for arm in s.ARMS:
  f=s.OUT/(arm+'_2100');mf=json.loads((f/'frozen_manifest.json').read_text());p=json.loads((f/'offline_predictions_frozen.json').read_text());assert mf['completed_epochs']==100 and mf['optimizer_steps']==1800 and not p['target_GT_scored'] and not p['decoder_used'];rs[arm]=p['records'];hh[arm]=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];computes[arm]={'training_parameters':387883,'inference_parameters':378011,'train_seconds':mf['train_seconds'],'inference_grid_seconds':p['seconds'],'auxiliary_forwards':1800,'main_forwards':1800}
 for field in ('raw_companion_sha256','mask_sha256','batch_stream_sha256','original_rng_stream_sha256'):
  assert all(a[field]==b[field] for a,b in zip(hh['E1'],hh['E2'])),field
 for arm,r in rs.items():assert [x['epoch'] for x in r]==s.PLAN['grid'];assert all(s.sha(x['path'])==x['sha256'] for x in r)
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all3groups10checkpoint_predictions_frozen':True,'E1_E2_main_raw_mask_RNG_streams_identical':True,'target_GT_no_training_or_hyperparameters':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());yp=cache/'data/target_labels.npy';assert s.sha(yp)==prov['data_sha256']['target_labels.npy'];labels=np.load(yp);assert len(labels)==52901;helper=s.module('completion_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];cls=[]
 for arm,records in rs.items():
  ev=[]
  for r in records:
   p=np.load(r['path']);ev.append({'epoch':r['epoch'],'metrics':helper.metrics(labels,p['predictions'],p['logits'],7)})
  s.dump(s.OUT/(arm+'_scored_grid.json'),ev)
  for ep,r in [('fixed100',ev[-1]),('target_best',max(ev,key=lambda x:x['metrics']['oa_percent']))]:
   v=r['metrics'];ends.append({'arm':arm,'seed':2100,'endpoint':ep,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa'],'zero_recall_classes':[i+1 for i,x in enumerate(v['per_class_percent']) if x==0],'collapse95':max(v['prediction_shares'])>=.95,'prediction_shares':v['prediction_shares']});cls.extend([{'arm':arm,'endpoint':ep,'class':i+1,'recall':x,'prediction_share':v['prediction_shares'][i]} for i,x in enumerate(v['per_class_percent'])])
 paired=[]
 for ep in ('fixed100','target_best'):
  rr={r['arm']:r for r in ends if r['endpoint']==ep}
  for arm,ref in [('E2','E0'),('E2','E1'),('E1','E0')]:paired.append({'endpoint':ep,'candidate':arm,'baseline':ref,**{k+'_delta':rr[arm][k]-rr[ref][k] for k in ('OA','AA','Kappa')}})
 fixed=[r for r in paired if r['endpoint']=='fixed100' and r['candidate']=='E2'];positive=all(r['OA_delta']>0 and r['AA_delta']>0 for r in fixed);s.dump(s.OUT/'summary.json',{'endpoints':ends,'paired':paired,'compute':computes,'source_only_calibration':json.loads((s.OUT/'source_only_calibration.json').read_text()),'fixed100_both_comparisons_OA_AA_positive':positive,'classification_coverage_review_required':True,'single_seed_exploratory_not_stability_evidence':True,'targets_scaled_using_source_only_statistics':True,'extra_source_target_supervision_not_target_effect_isolated':True,'target_best_target_oracle_diagnostic':True,'no_auto_followup':True})
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',cls)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 s.verify();s.dump(s.OUT/'status.json',{'status':'completed_two_exploratory_trainings_and_offline_scoring','auto_followup':False,'new_benchmark':False});print('COMPLETION_REPORT_FINISHED',json.dumps(ends),flush=True)
if __name__=='__main__':main()
