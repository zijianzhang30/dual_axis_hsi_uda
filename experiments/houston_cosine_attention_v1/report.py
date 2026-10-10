import csv,json
import numpy as np
import study as s

def main():
 s.verify();rs={'N0':[r for r in json.loads((s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json').read_text())['records'] if r['epoch']<=100]};comp={};hist={}
 for arm in s.ARMS:
  folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());p=json.loads((folder/'offline_predictions_frozen.json').read_text());assert mf['completed_epochs']==100 and not p['target_GT_scored'];rs[arm]=p['records'];hist[arm]=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()];comp[arm]={'parameters':378019,'added_parameters':8,'fixed_scale_buffer_elements':8 if arm=='N2' else 0,'train_seconds':mf['train_seconds'],'evaluation_seconds':p['seconds'],'source_val_final':mf['source_val_final']}
 assert all(a['batch_stream_sha256']==b['batch_stream_sha256'] and a['original_rng_stream_sha256']==b['original_rng_stream_sha256'] for a,b in zip(hist['N1'],hist['N2']))
 for rr in rs.values():assert [r['epoch'] for r in rr]==s.PLAN['grid'] and all(s.sha(r['path'])==r['sha256'] for r in rr)
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all_predictions_frozen_before_GT_scoring':True,'no_target_label_parameter_selection':True,'calibration_source_only':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());lp=cache/'data/target_labels.npy';assert s.sha(lp)==prov['data_sha256']['target_labels.npy'];y=np.load(lp);helper=s.module('cosine_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];cls=[];attrows=[]
 for arm,rr in rs.items():
  grid=[]
  for r in rr:
   p=np.load(r['path']);grid.append({'epoch':r['epoch'],'metrics':helper.metrics(y,p['predictions'],p['logits'],7),'attention_diagnostics':r.get('first128_eval_attention_diagnostics')})
  s.dump(s.OUT/(arm+'_scored_grid.json'),grid)
  for endpoint,r in [('fixed100',grid[-1]),('target_best',max(grid,key=lambda r:r['metrics']['oa_percent']))]:
   a=r['metrics'];ends.append({'arm':arm,'seed':2100,'endpoint':endpoint,'epoch':r['epoch'],'OA':a['oa_percent'],'AA':a['aa_percent'],'Kappa':a['kappa'],'zero_recall_classes':[i+1 for i,z in enumerate(a['per_class_percent']) if z==0],'collapse95':max(a['prediction_shares'])>=.95,'prediction_shares':a['prediction_shares']});cls.extend([{'arm':arm,'endpoint':endpoint,'class':i+1,'recall':z,'prediction_share':a['prediction_shares'][i]} for i,z in enumerate(a['per_class_percent'])])
  if arm!='N0':
   for row in hist[arm]:
    d=row['first_batch_diagnostics']['attention'];g=row['first_batch_diagnostics']['head_log_scale_gradients']
    for h in range(8):attrows.append({'arm':arm,'epoch':row['epoch'],'head':h,'effective_scale_after_epoch':row['head_effective_scales'][h],'log_scale_after_epoch':row['head_log_scale_parameters'][h],'firstbatch_scale_gradient':g[h],'firstbatch_attention_entropy':d['head_attention_entropy'][h],'firstbatch_score_RMS':d['head_row_centered_score_RMS'][h],'firstbatch_Q_norm':d['head_Q_norm_mean'][h],'firstbatch_K_norm':d['head_K_norm_mean'][h]})
 paired=[]
 for ep in ('fixed100','target_best'):
  rows={r['arm']:r for r in ends if r['endpoint']==ep}
  for a,b in [('N2','N0'),('N2','N1'),('N1','N0')]:paired.append({'endpoint':ep,'candidate':a,'baseline':b,**{k+'_delta':rows[a][k]-rows[b][k] for k in ('OA','AA','Kappa')}})
 positive=all(r['OA_delta']>0 and r['AA_delta']>0 for r in paired if r['endpoint']=='fixed100' and r['candidate']=='N2');summary={'endpoints':ends,'paired':paired,'compute':comp,'fixed100_both_comparisons_OA_AA_positive':positive,'single_seed_not_stability_evidence':True,'target_best_GT_oracle_diagnostic':True,'source_only_calibration':json.loads((s.OUT/'source_calibration.json').read_text()),'no_auto_followup':True};s.dump(s.OUT/'summary.json',summary)
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',cls),('head_diagnostics.csv',attrows)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 lines=['Houston first-layer scaledcosineattention, seed2100,fixed100 exploratory','N0 formalC reuse;N1 dotproduct xexp(a);N2 cosine xsourcecalibratedfixed_s_h0 xexp(a); allsourceCEonly','Fixed endpoint and target-label-selected best on identical10:10:100grid are separate. No +/-std for one seed.']
 for r in ends:lines.append('%s %s epoch%d OA %.6f AA %.6f Kappa %.6f zero-recall%s collapse95%s'%(r['arm'],r['endpoint'],r['epoch'],r['OA'],r['AA'],r['Kappa'],r['zero_recall_classes'],r['collapse95']))
 for r in paired:lines.append('%s %s-%s OA %+.6f AA %+.6f Kappa %+.6f'%(r['endpoint'],r['candidate'],r['baseline'],r['OA_delta'],r['AA_delta'],r['Kappa_delta']))
 lines.extend(['Screen: positive fixed100 signal requiring classcoverage review,not stabilityproof.' if positive else 'No-Go under locked fixed100 OA/AA screen; best-onlysignals reported separately,noautomaticfollowup.','N1/N2 each8newtrainableparameters,378011->378019. Only first source/targetselfattention score changed; originalunusedcoupled calculations remainforRNG, no coupled objective.','N1 initial logits,basegradient,one-stepbaseupdates equalN0; N2 is structuralchange,not requiredinitialoutputequal. Sourcecalibrationmatches rowcenteredscoreRMS only,not attentiondistribution/gradients.','Calibration 1024sourceTRAINsamples,eval/noaug/frozenBN;FP64globalRMS; denominator>=1e-8 andpositivescales verified; N2 has NOadditional sqrt divisor. Fixeds_h0 savedbuffer and8logscales checkpointed.','All100epochs main data/RNG hashes matchN0,1800updates,normalBN/Transformercalls unchanged. Targetsclasslabels only read afterpredictionbarrier; inheritednonzeroGTsamplingmask disclosed.','This is transfer of existingcosineattention mechanism,not an establishedoriginalcontribution. No TalkingHeads, furtherseeds,benchmarkexpansion or temperaturetuning launched.','Newtraining walltimes are measured under sharedGPUloads,not controlledpureoverhead estimates.'])
 for arm in s.ARMS:lines.append(arm+' final headscales '+str(hist[arm][-1]['head_effective_scales'])+' sourceval '+str(comp[arm]['source_val_final'])+' trainseconds '+str(comp[arm]['train_seconds']))
 (s.OUT/'CONCLUSION.txt').write_text('\n'.join(lines)+'\n');s.verify();s.dump(s.OUT/'status.json',{'status':'completed_two_exploratory_trainings_and_offline_scoring','auto_followup':False});print('COSINE_REPORT_COMPLETE',json.dumps(ends),flush=True)
if __name__=='__main__':main()
