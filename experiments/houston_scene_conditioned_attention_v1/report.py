import csv,json
import numpy as np
import study as s

def main():
 s.verify();rs={'S0':[r for r in json.loads((s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json').read_text())['records'] if r['epoch']<=100]};comp={};hist={}
 for arm in s.ARMS:
  folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());p=json.loads((folder/'offline_predictions_frozen.json').read_text());assert mf['completed_epochs']==100 and not p['target_GT_scored'];rs[arm]=p['records'];hist[arm]=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()];comp[arm]={'parameters':384251,'added_parameters':6240,'train_seconds':mf['train_seconds'],'evaluation_seconds':p['seconds'],'source_val_final':mf['source_val_final']}
 assert all(a['batch_stream_sha256']==b['batch_stream_sha256'] and a['original_rng_stream_sha256']==b['original_rng_stream_sha256'] for a,b in zip(hist['S1'],hist['S2']))
 for rr in rs.values():assert [r['epoch'] for r in rr]==s.PLAN['grid'] and all(s.sha(r['path'])==r['sha256'] for r in rr)
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all_predictions_frozen_before_GT_scoring':True,'no_target_label_parameter_selection':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());lp=cache/'data/target_labels.npy';assert s.sha(lp)==prov['data_sha256']['target_labels.npy'];y=np.load(lp);helper=s.module('scene_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];cls=[]
 for arm,rr in rs.items():
  grid=[]
  for r in rr:
   p=np.load(r['path']);grid.append({'epoch':r['epoch'],'metrics':helper.metrics(y,p['predictions'],p['logits'],7),'diagnostics':r.get('diagnostics')})
  s.dump(s.OUT/(arm+'_scored_grid.json'),grid)
  for endpoint,r in [('fixed100',grid[-1]),('target_best',max(grid,key=lambda r:r['metrics']['oa_percent']))]:
   a=r['metrics'];ends.append({'arm':arm,'seed':2100,'endpoint':endpoint,'epoch':r['epoch'],'OA':a['oa_percent'],'AA':a['aa_percent'],'Kappa':a['kappa'],'zero_recall_classes':[i+1 for i,z in enumerate(a['per_class_percent']) if z==0],'collapse95':max(a['prediction_shares'])>=.95,'prediction_shares':a['prediction_shares']});cls.extend([{'arm':arm,'endpoint':endpoint,'class':i+1,'recall':z,'prediction_share':a['prediction_shares'][i]} for i,z in enumerate(a['per_class_percent'])])
 paired=[]
 for ep in ('fixed100','target_best'):
  rows={r['arm']:r for r in ends if r['endpoint']==ep}
  for a,b in [('S2','S0'),('S2','S1'),('S1','S0')]:paired.append({'endpoint':ep,'candidate':a,'baseline':b,**{k+'_delta':rows[a][k]-rows[b][k] for k in ('OA','AA','Kappa')}})
 positive=all(r['OA_delta']>0 and r['AA_delta']>0 for r in paired if r['endpoint']=='fixed100' and r['candidate']=='S2');summary={'endpoints':ends,'paired':paired,'compute':comp,'fixed100_both_comparisons_OA_AA_positive':positive,'single_seed_not_stability_evidence':True,'target_best_GT_oracle_diagnostic':True,'no_auto_followup':True};s.dump(s.OUT/'summary.json',summary)
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',cls)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 lines=['Houston scene-conditioned first self-attention, seed2100, fixed100 exploratory','S0 frozen formal C; S1 source summary; S2 target summary; source CE only','Fixed endpoint and target-label-selected best on identical10:10:100 grid are separate. No +/-std for one seed.']
 for r in ends:lines.append('%s %s epoch%d OA %.6f AA %.6f Kappa %.6f zero-recall%s collapse95%s'%(r['arm'],r['endpoint'],r['epoch'],r['OA'],r['AA'],r['Kappa'],r['zero_recall_classes'],r['collapse95']))
 for r in paired:lines.append('%s %s-%s OA %+.6f AA %+.6f Kappa %+.6f'%(r['endpoint'],r['candidate'],r['baseline'],r['OA_delta'],r['AA_delta'],r['Kappa_delta']))
 lines.extend(['Screen: positive fixed100 signal requiring class-coverage review.' if positive else 'No-Go under locked fixed100 OA/AA screen; no automatic followup.','No evidence of stability/novelty from single seed. Only first self attention changed; internal V projections, FFN, tokenizer, CNN and original unused coupled calculations retained.','Each S1/S2 adds6240 trainableparams (378011->384251). Frozen1024sample source/target cohort used for saved current-model summaries; additional eval stem forwards reported, noBNupdates. Training sourceval schedule and all100 data/main RNG hashes reproduce S0.','S1 eval batch alternative uses target batch, so that diagnostic changes context domain as well as fixed-vs-batch. S2 comparison isolates its target-summary batch/fixed mismatch.','Training is target-aware, with inherited GT-nonzero candidate mask; target class labels only read after prediction barrier.','No retraining, tuning, extra seeds or benchmark expansion launched.'])
 (s.OUT/'CONCLUSION.txt').write_text('\n'.join(lines)+'\n');s.verify();s.dump(s.OUT/'status.json',{'status':'completed_two_exploratory_trainings_and_offline_scoring','auto_followup':False});print('SCENE_REPORT_COMPLETE',json.dumps(ends),flush=True)
if __name__=='__main__':main()
