import csv,json
import numpy as np
import study as s

def main():
 s.verify();rs={'TH0':[r for r in json.loads((s.B/'houston_shared_adapter_2x2_v1/CE0_2100/offline_predictions_frozen.json').read_text())['records'] if r['epoch']<=100]};comp={};hist={}
 for arm in s.ARMS:
  folder=s.OUT/(arm+'_2100');cfg=json.loads((folder/'config.json').read_text());mf=json.loads((folder/'frozen_manifest.json').read_text());p=json.loads((folder/'offline_predictions_frozen.json').read_text());assert mf['completed_epochs']==100 and not p['target_GT_scored'];rs[arm]=p['records'];hist[arm]=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()];comp[arm]={'parameters':cfg['parameters'],'added_parameters':cfg['install_audit']['added_parameters'],'train_seconds':mf['train_seconds'],'evaluation_seconds':p['seconds'],'source_val_final':mf['source_val_final']}
 assert all(a['batch_stream_sha256']==b['batch_stream_sha256'] and a['original_rng_stream_sha256']==b['original_rng_stream_sha256'] for a,b in zip(hist['TH1'],hist['TH2']))
 for rr in rs.values():assert [r['epoch'] for r in rr]==s.PLAN['grid'] and all(s.sha(r['path'])==r['sha256'] for r in rr)
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all_predictions_frozen_before_GT_scoring':True,'no_target_label_parameter_selection':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());lp=cache/'data/target_labels.npy';assert s.sha(lp)==prov['data_sha256']['target_labels.npy'];y=np.load(lp);helper=s.module('talking_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];cls=[];attentionrows=[]
 for arm,rr in rs.items():
  grid=[]
  for r in rr:
   p=np.load(r['path']);grid.append({'epoch':r['epoch'],'metrics':helper.metrics(y,p['predictions'],p['logits'],7),'attention_diagnostics':r.get('first128_eval_attention_diagnostics')})
  s.dump(s.OUT/(arm+'_scored_grid.json'),grid)
  for endpoint,r in [('fixed100',grid[-1]),('target_best',max(grid,key=lambda r:r['metrics']['oa_percent']))]:
   a=r['metrics'];ends.append({'arm':arm,'seed':2100,'endpoint':endpoint,'epoch':r['epoch'],'OA':a['oa_percent'],'AA':a['aa_percent'],'Kappa':a['kappa'],'zero_recall_classes':[i+1 for i,z in enumerate(a['per_class_percent']) if z==0],'collapse95':max(a['prediction_shares'])>=.95,'prediction_shares':a['prediction_shares']});cls.extend([{'arm':arm,'endpoint':endpoint,'class':i+1,'recall':z,'prediction_share':a['prediction_shares'][i]} for i,z in enumerate(a['per_class_percent'])])
  if arm!='TH0':
   for row in hist[arm]:
    d=row['first_batch_diagnostics']['attention'];g=row['first_batch_diagnostics']['new_parameter_gradients']
    for h in range(8):attentionrows.append({'arm':arm,'epoch':row['epoch'],'head':h,'pre_diagonal':d['talk_pre']['diagonal'][h],'post_diagonal':d['talk_post']['diagonal'][h],'pre_offdiag_Frobenius':d['talk_pre']['offdiagonal_Frobenius'],'post_offdiag_Frobenius':d['talk_post']['offdiagonal_Frobenius'],'pre_gradient_norm':g['talk_pre']['norm'],'post_gradient_norm':g['talk_post']['norm'],'pre_offdiag_gradient_norm':g['talk_pre']['offdiagonal_norm'],'post_offdiag_gradient_norm':g['talk_post']['offdiagonal_norm'],'softmax_entropy_BEFOREpostmix':d['head_softmax_entropy_BEFORE_postmix'][h],'postmix_row_mass_mean':d['head_postmix_row_mass_mean'][h],'postmix_negative_weight_fraction':d['postmix_negative_weight_fraction']})
 paired=[]
 for ep in ('fixed100','target_best'):
  rows={r['arm']:r for r in ends if r['endpoint']==ep}
  for a,b in [('TH2','TH0'),('TH2','TH1'),('TH1','TH0')]:paired.append({'endpoint':ep,'candidate':a,'baseline':b,**{k+'_delta':rows[a][k]-rows[b][k] for k in ('OA','AA','Kappa')}})
 positive=all(r['OA_delta']>0 and r['AA_delta']>0 for r in paired if r['endpoint']=='fixed100' and r['candidate']=='TH2');summary={'endpoints':ends,'paired':paired,'compute':comp,'fixed100_both_comparisons_OA_AA_positive':positive,'single_seed_not_stability_evidence':True,'not_parameter_matched':True,'TH2_extra_offdiagonal_parameters_over_TH1':112,'target_best_GT_oracle_diagnostic':True,'reference':json.loads((s.OUT/'code_snapshot/reference_provenance.json').read_text()),'no_auto_followup':True};s.dump(s.OUT/'summary.json',summary)
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',paired),('per_class_recall.csv',cls),('head_diagnostics.csv',attentionrows)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 lines=['Houston first-layer TalkingHeads,seed2100,fixed100 exploratory','TH0 originalC/native dotproduct;TH1 diagonalpre/post scaling;TH2 fullpre/post headmixing; allsourceCEonly','Fixed endpoint and target-label-selected best on identical10:10:100grid are separate. No +/-std for one seed.']
 for r in ends:lines.append('%s %s epoch%d OA %.6f AA %.6f Kappa %.6f zero-recall%s collapse95%s'%(r['arm'],r['endpoint'],r['epoch'],r['OA'],r['AA'],r['Kappa'],r['zero_recall_classes'],r['collapse95']))
 for r in paired:lines.append('%s %s-%s OA %+.6f AA %+.6f Kappa %+.6f'%(r['endpoint'],r['candidate'],r['baseline'],r['OA_delta'],r['AA_delta'],r['Kappa_delta']))
 lines.extend(['Screen: positivefixed100signal requiringclasscoverage review,not stabilityproof.' if positive else 'No-Go underlockedfixed100 OA/AA screen;best-onlysignals separate,noautomaticfollowup.','TH1adds16/TH2adds128trainableparams:378027/378139. Fullheadmixinghas112extraoffdiagonalweights;not strictcapacitymatched. No placeholderparams.','Bothpre/post effectiveheadmatrices startidentity,no biases. Initialall4outputs/basegradients/baseone-step updates reproduceTH0;newmixingparameters receivegradient,TH2offdiagonals receivegradientandupdate.','Only firstsource/targetselfattention changed. Originalunusedcoupledcalculations/dropoutorder preserved;originalsharedQKV/V/FFN/CLS/classifier unchanged.','Postsoftmaxlinearheadprojection is not forcednonnegative/row-normalized; negativeweights/rowmasslogged. Entropydefinedonlyon softmaxBEFOREpostmix. No newactivationor secondsoftmax.','All100main data/RNG hashes matchTH0,1800updates,BN/Transformercalls unchanged; no extraauxforward. InheritedtargetnonzeroGTcandidate mask disclosed;targetclasslabels read onlyafterpredictionbarrier.','TalkingHeads isexistingmechanism,not establishedoriginalinnovation. No otherstructure/loss/seed/benchmark or sweep automaticallystarted.','TimingunderdifferentsharedGPUloads isnot acontrolledpureoverheadcomparison.'])
 for arm in s.ARMS:
  d=hist[arm][-1]['first_batch_diagnostics']['attention'];lines.append(arm+' finalfirstbatch preoffdiag '+str(d['talk_pre']['offdiagonal_Frobenius'])+' postoffdiag '+str(d['talk_post']['offdiagonal_Frobenius'])+' negativeweights '+str(d['postmix_negative_weight_fraction'])+' sourceval '+str(comp[arm]['source_val_final'])+' trainseconds '+str(comp[arm]['train_seconds']))
 (s.OUT/'CONCLUSION.txt').write_text('\n'.join(lines)+'\n');s.verify();s.dump(s.OUT/'status.json',{'status':'completed_two_exploratory_trainings_and_offline_scoring','auto_followup':False});print('TALKING_REPORT_COMPLETE',json.dumps(ends),flush=True)
if __name__=='__main__':main()
