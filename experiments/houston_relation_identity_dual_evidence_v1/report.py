import csv,json,numpy as np
import study as s

def write(name,rows):
 with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 s.verify();barrier=json.loads((s.OUT/'ALL_TARGET_EVIDENCE_FROZEN_BEFORE_GT.json').read_text());assert barrier['target_GT_not_scored_in_this_study'];records={r['router']:r for r in barrier['records']}
 for r in records.values():assert s.sha(r['path'])==r['sha256']
 assert s.sha(barrier['agreement_path'])==barrier['agreement_sha256']
 h=json.loads((s.OUT/'training_complete.json').read_text());assert h['epochs']==100 and h['updates']==1800 and h['all100_source_aug_data_RNG_hashes_match_R0'] and h['relation_BN_updates']==0
 # Only now load target labels for offline scoring of this experiment.
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());yp=cache/'data/target_labels.npy';cp=cache/'data/target_coords.npy';assert s.sha(yp)==prov['data_sha256']['target_labels.npy'] and s.sha(cp)==prov['data_sha256']['target_coords.npy'];y=np.load(yp);coords=np.load(cp);helper=s.module('dual_evidence_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');pred={};metric={};ends=[];pc=[]
 for k,r in records.items():
  z=np.load(r['path']);assert np.array_equal(z['coordinates'],coords);pred[k]=z['predictions'];m=helper.metrics(y,pred[k],z['logits'],7);metric[k]=m;ends.append({'router':k,'seed':2100,'epoch':100,'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],'zero_recall_classes':[i+1 for i,v in enumerate(m['per_class_percent']) if v==0],'collapse95':max(m['prediction_shares'])>=.95,'max_prediction_share':max(m['prediction_shares'])})
  pc.extend([{'router':k,'class':i+1,'n':int((y==i).sum()),'recall':x,'delta_vs_R0_pp':x-metric['R0']['per_class_percent'][i] if k!='R0' else 0.,'prediction_share':m['prediction_shares'][i]} for i,x in enumerate(m['per_class_percent'])])
 agreement=np.load(barrier['agreement_path']);agree=pred['R0']==pred['R1'];assert np.array_equal(agree,agreement['agreement']) and np.array_equal(~agree,agreement['disagreement']);old=pred['R0']==y;rec=[]
 for k in ('R1','R2','R3'):
  good=pred[k]==y;recovered=int((~old&good).sum());damaged=int((old&~good).sum());assert recovered-damaged==int(good.sum())-int(old.sum())
  rec.append({'router':k,'R0_errors':int((~old).sum()),'recovered_count':recovered,'recovery_rate_percent':recovered/(~old).sum()*100,'R0_correct':int(old.sum()),'damaged_count':damaged,'damage_rate_percent':damaged/old.sum()*100,'OA_delta_pp':metric[k]['oa_percent']-metric['R0']['oa_percent'],'AA_delta_pp':metric[k]['aa_percent']-metric['R0']['aa_percent'],'Kappa_delta':metric[k]['kappa']-metric['R0']['kappa']})
 subsets=[]
 for group,mask in [('all',np.ones(len(y),bool)),('agreement',agree),('disagreement',~agree)]:
  for c in [None]+list(range(7)):
   use=mask if c is None else mask&(y==c);n=int(use.sum());row={'subset':group,'true_class':'all' if c is None else c+1,'n':n,'target_fraction_percent':n/len(y)*100}
   for k in ('R0','R1','R2','R3'):row[k+'_correct']=int((use&(pred[k]==y)).sum());row[k+'_accuracy_percent']=row[k+'_correct']/n*100 if n else float('nan')
   row['either_expert_correct_percent']=float(((pred['R0']==y)|(pred['R1']==y))[use].mean()*100) if n else float('nan');subsets.append(row)
 # Disagreement outcomes and predicted-pair accuracy, purely descriptive.
 modes=[]
 for rc in (False,True):
  for sc in (False,True):
   mask=(~agree)&((pred['R0']==y)==rc)&((pred['R1']==y)==sc);modes.append({'relation_correct':rc,'spectral_correct':sc,'n':int(mask.sum()),'disagreement_fraction_percent':float(mask.sum()/(~agree).sum()*100)})
 assert sum(r['n'] for r in modes)==int((~agree).sum());assert not next(r['n'] for r in modes if r['relation_correct'] and r['spectral_correct'])
 pairs=[]
 for a in range(7):
  for b in range(7):
   if a==b:continue
   mask=(pred['R0']==a)&(pred['R1']==b);n=int(mask.sum());pairs.append({'R0_pred':a+1,'R1_pred':b+1,'n':n,**{k+'_accuracy_percent':float((pred[k][mask]==y[mask]).mean()*100) if n else float('nan') for k in ('R0','R1','R2','R3')}})
 perrecovery=[]
 for c in range(7):
  mask=y==c;bad=mask&~old;good=mask&old
  for k in ('R2','R3'):
   now=pred[k]==y;perrecovery.append({'router':k,'class':c+1,'R0_errors':int(bad.sum()),'recovered':int((bad&now).sum()),'recovery_percent':float((bad&now).sum()/bad.sum()*100) if bad.any() else float('nan'),'R0_correct':int(good.sum()),'damaged':int((good&~now).sum()),'damage_percent':float((good&~now).sum()/good.sum()*100) if good.any() else float('nan')})
 by={r['router']:r for r in ends};deltas={r['router']:r for r in rec};gate={}
 for k in ('R2','R3'):
  d=deltas[k];gate[k]={'OA_delta_at_least_0_5pp':bool(d['OA_delta_pp']>=.5),'AA_delta_at_least_minus_0_5pp':bool(d['AA_delta_pp']>=-.5),'no_new_zero_recall':not set(by[k]['zero_recall_classes'])-set(by['R0']['zero_recall_classes']),'no_new_collapse':not by[k]['collapse95'] or by['R0']['collapse95']};gate[k]['passed']=all(gate[k].values())
 passing=[k for k,v in gate.items() if v['passed']];go=bool(passing);summary={'decision':'Go' if go else 'No-Go','passing_fixed_rules':passing,'conditions':gate,'endpoints':ends,'recovery_damage_and_deltas':rec,'expert_agreement_counts':{'agree':int(agree.sum()),'disagree':int((~agree).sum())},'disagreement_outcomes':modes,'subset_expert_accuracy':subsets,'per_class':pc,'spectral_parameters':3687,'relation_expert_checkpoint_and_gradients_unchanged':True,'same_source_aug_batches_and_global_RNG_all100_epochs':True,'no_target_label_in_training_or_fusion':True,'previous_Houston_metrics_already_known':True,'no_followup_started':True,'single_seed_minimal_validation_only':True,'stop_MoE_gate_fusion_variants_if_NoGo':not go};s.dump(s.OUT/'summary.json',summary);s.dump(s.OUT/'scored_full_metrics.json',metric)
 for name,rr in [('fixed100_metrics.csv',ends),('per_class_recall.csv',pc),('recovery_damage_and_deltas.csv',rec),('expert_agreement_disagreement_accuracy.csv',subsets),('disagreement_outcomes.csv',modes),('disagreement_predicted_pair_accuracy.csv',pairs),('per_class_recovery_damage.csv',perrecovery)]:write(name,rr)
 ds=next(r for r in subsets if r['subset']=='disagreement' and r['true_class']=='all');lines=['houston_relation_identity_dual_evidence_v1','判定：'+summary['decision'],'Houston seed2100 / fixed100 / 独立Spectral Identity Expert源域CE。','R0完整复用原C固定epoch100，不重训、不改参数/BN/梯度；R1只输入中心48-band光谱，LN48→Linear48to64→GELU→Linear64to7，3687参数。','R2固定0.5概率平均；R3固定归一化概率乘积；全部temperature=1。','','router  OA%  AA%  Kappa']
 for r in ends:lines.append('%s %.4f %.4f %.6f'%(r['router'],r['OA'],r['AA'],r['Kappa']))
 for r in rec:lines.append('%s-R0: OA%+.4fpp / AA%+.4fpp / Kappa%+.6f; recovered%d (%.4f%% of errors), damaged%d (%.4f%% of correct)'%(r['router'],r['OA_delta_pp'],r['AA_delta_pp'],r['Kappa_delta'],r['recovered_count'],r['recovery_rate_percent'],r['damaged_count'],r['damage_rate_percent']))
 lines+=['','预锁定Go条件：'+json.dumps(gate),'zero recall：'+json.dumps({r['router']:r['zero_recall_classes'] for r in ends}),'collapse95：'+json.dumps({r['router']:r['collapse95'] for r in ends}),'disagreement n=%d, R0 accuracy %.4f%% / R1 accuracy %.4f%%'%(ds['n'],ds['R0_accuracy_percent'],ds['R1_accuracy_percent']),'','所有target logits/predictions/coordinates和两expert agreement/disagreement先保存并hash，随后读取target GT评分。','原C每个source batch/augmentation与global RNG流100轮全部重放一致，relation本轮0次forward和0次BN更新；spectral无stochastic层，独立梯度及优化器。','不使用target pseudo-label/GT训练，不用target选择checkpoint、融合公式、权重、温度或阈值；此前R0和其他Houston实验指标已知，不宣称全新未触及测试。','仅fixed100，不做target-best选模。未追加任何训练。'+('融合规则未通过，按约定停止MoE/learned gate/更多融合规则。' if not go else '至少一个固定融合规则通过；未启动后续实验，需要另行讨论。')]
 (s.OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n');(s.OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n\nLOCKED PLAN\n'+json.dumps(s.PLAN,indent=2)+'\n\nFULL SUMMARY\n'+json.dumps(summary,indent=2)+'\n\nPER CLASS\n'+(s.OUT/'per_class_recall.csv').read_text()+'\n\nAGREEMENT / DISAGREEMENT\n'+(s.OUT/'expert_agreement_disagreement_accuracy.csv').read_text());s.verify();s.dump(s.OUT/'FINAL_REVIEW.json',{'decision':summary['decision'],'source_only_spectral_training_100_epochs_complete':True,'relation_state_unchanged':True,'no_backbone_gradient_or_BN_update':True,'all_predictions_and_disagreement_frozen_before_scoring':True,'fixed_rules_applied_without_tuning':True,'no_new_experiment_started':True});print('DUAL_EVIDENCE_REPORT_COMPLETE',summary['decision'],json.dumps(ends),json.dumps(ds),flush=True)
if __name__=='__main__':main()
