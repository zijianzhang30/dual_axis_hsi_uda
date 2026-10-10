import csv,json,numpy as np
import study as s

def main():
 s.verify();folder=s.OUT/'S3_2100';mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==100 and mf['data_RNG_all_epochs_match_original_C'];files={}
 for a,f in [('S0',s.previous()/'S0_2100'),('S3',folder)]:
  q=json.loads((f/'offline_predictions_frozen.json').read_text());assert q['target_GT_not_scored'] and q['model_and_running_stats_unchanged'];files[a]=q['records'];assert [r['epoch'] for r in files[a]]==s.PLAN['grid']
  for r in files[a]:assert s.sha(r['path'])==r['sha256']
 s.dump(s.OUT/'ALL_PREDICTIONS_FROZEN_BEFORE_GT.json',{'records':files,'S3_all10_predictions_frozen_before_scoring':True,'S0_reused_old_frozen_predictions':True,'previous_S1_S2_results_already_known':True,'new_S3_configuration_and_Go_rules_locked_before_training':True,'no_target_GT_in_S3_loss_or_checkpoint_selection':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());yp=cache/'data/target_labels.npy';cp=cache/'data/target_coords.npy';assert s.sha(yp)==prov['data_sha256']['target_labels.npy'] and s.sha(cp)==prov['data_sha256']['target_coords.npy'];y=np.load(yp);coords=np.load(cp);helper=s.module('S3_offline_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];pc=[];grid=[];pred={};metrics={}
 for a,rs in files.items():
  ev=[]
  for r in rs:
   z=np.load(r['path']);assert np.array_equal(z['coordinates'],coords);v=helper.metrics(y,z['predictions'],z['logits'],7);ev.append({'epoch':r['epoch'],'metrics':v});grid.append({'arm':a,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa']})
   if r['epoch']==100:pred[a]=z['predictions'].copy();metrics[a]=v
  s.dump(s.OUT/(a+'_scored_grid.json'),ev)
  for endpoint,r in [('fixed100',ev[-1]),('same_grid_best_diagnostic',max(ev,key=lambda x:x['metrics']['oa_percent']))]:
   v=r['metrics'];ends.append({'arm':a,'seed':2100,'endpoint':endpoint,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa'],'zero_recall_classes':[i+1 for i,x in enumerate(v['per_class_percent']) if x==0],'collapse95':max(v['prediction_shares'])>=.95,'max_prediction_share':max(v['prediction_shares'])})
   pc.extend([{'arm':a,'endpoint':endpoint,'class':i+1,'recall':x,'prediction_share':v['prediction_shares'][i]} for i,x in enumerate(v['per_class_percent'])])
 old=pred['S0']==y;new=pred['S3']==y;per=[]
 for c in range(7):
  mask=y==c;bad=mask&~old;good=mask&old
  per.append({'class':c+1,'n':int(mask.sum()),'S0_recall':metrics['S0']['per_class_percent'][c],'S3_recall':metrics['S3']['per_class_percent'][c],'delta_pp':metrics['S3']['per_class_percent'][c]-metrics['S0']['per_class_percent'][c],'recovered_count':int((bad&new).sum()),'damaged_count':int((good&~new).sum()),'S0_errors':int(bad.sum()),'S0_correct':int(good.sum())})
 delta=[]
 for endpoint in ('fixed100','same_grid_best_diagnostic'):
  rr={r['arm']:r for r in ends if r['endpoint']==endpoint};delta.append({'endpoint':endpoint,**{k+'_delta':rr['S3'][k]-rr['S0'][k] for k in ('OA','AA','Kappa')},'S3_epoch':rr['S3']['epoch'],'S0_epoch':rr['S0']['epoch']})
 fixed={r['arm']:r for r in ends if r['endpoint']=='fixed100'};d=delta[0];damaged_classes=[r['class'] for r in per if r['delta_pp']<=-5];gate={'OA_delta_strictly_above_0_5pp':d['OA_delta']>.5,'AA_delta_at_least_minus_0_5pp':d['AA_delta']>=-.5,'no_new_zero_recall':not set(fixed['S3']['zero_recall_classes'])-set(fixed['S0']['zero_recall_classes']),'no_new_collapse':not fixed['S3']['collapse95'] or fixed['S0']['collapse95'],'no_class_recall_drop_ge_5pp':not damaged_classes};go=all(gate.values());recovery={'S0_errors':int((~old).sum()),'recovered':int((~old&new).sum()),'recovery_rate_percent':float((~old&new).sum()/(~old).sum()*100),'S0_correct':int(old.sum()),'damaged':int((old&~new).sum()),'damage_rate_percent':float((old&~new).sum()/old.sum()*100)};assert recovery['recovered']-recovery['damaged']==int(new.sum())-int(old.sum())
 hist=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()];assert len(hist)==100;reference=[json.loads(x) for x in (s.previous()/'S0_2100/history.jsonl').read_text().splitlines()]
 for i,r in enumerate(hist):
  assert r['epoch']==i+1 and r['steps']==18 and all(v==18*(i+1) for v in r['BN_counts'].values());assert r['main_forward_calls']=={'block_0':18,'block_1':18,'block_2':18,'classifier':72}
  assert all(r[k]==reference[i][k] for k in ('batch_stream_sha256','original_rng_stream_sha256'))
 assert hist[0]['first_batch_gradients']['encoder_weight_grad_norm']==0 and hist[0]['first_batch_gradients']['alpha_grad'][0]!=0;assert all(r['first_batch_gradients']['encoder_weight_grad_norm']>0 for r in hist[1:])
 summary={'decision':'Go' if go else 'No-Go','Go_conditions':gate,'damage_classes_drop_ge_5pp':damaged_classes,'endpoints':ends,'deltas':delta,'per_class':per,'fixed100_recovery_damage':recovery,'final_alpha':hist[-1]['alpha_after_epoch'],'parameters':json.loads((folder/'config.json').read_text())['parameters'],'initial_preflight':json.loads((s.OUT/'preflight.json').read_text()),'same_seed_all100_data_RNG_BN_calls_match_S0':True,'relation_tokens_not_directly_injected':True,'backbone_trainable_not_frozen':True,'one_new_training_only':True,'no_followups_started':True,'spectral_gate_token_direction_stopped':not go,'single_seed_screening_not_cross_seed_validation':True,'causal_relation_pollution_not_proven_by_this_comparison':True};s.dump(s.OUT/'summary.json',summary)
 for name,rr in [('endpoints.csv',ends),('paired_deltas.csv',delta),('per_class_recall.csv',pc),('per_class_fixed100_deltas.csv',per),('same_grid_diagnostic.csv',grid)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 lines=['houston_final_cls_spectral_anchor_v1 / S3','判定：'+summary['decision'],'仅新增一组 Houston seed2100 / source CE only / fixed100；S0复用上一轮同seed固定100轮。','注入点：最后一个Transformer block之后、原最终LayerNorm之前；仅修改CLS，后续无attention将anchor传播到relation tokens。','编码器/gate沿用S2形式及初值，Wv为恒等，alpha=0；主网络仍正常训练，并非冻结所有关系参数。','','arm  OA%  AA%  Kappa']
 for a in ('S0','S3'):r=fixed[a];lines.append('%s %.4f %.4f %.6f'%(a,r['OA'],r['AA'],r['Kappa']))
 lines+=['S3-S0: OA %+.4fpp / AA %+.4fpp / Kappa %+.6f'%(d['OA_delta'],d['AA_delta'],d['Kappa_delta']),'锁定门槛：'+json.dumps(gate),'recall下降达到5pp的类：'+json.dumps(damaged_classes),'zero recall：'+json.dumps({a:r['zero_recall_classes'] for a,r in fixed.items()}),'collapse95：'+json.dumps({a:r['collapse95'] for a,r in fixed.items()}),'原错误恢复/原正确破坏：'+json.dumps(recovery),'','零注入logits/原梯度/SGD更新/RNG/BN逐位等价通过；非零注入直接保持T1..T4逐位不变。alpha首批有非零梯度，encoder后续有非零梯度。','所有S3预测及坐标先冻结并hash，再读取target GT评分。旧S0/S1/S2指标此前已知，本轮是独立预锁定的一个位置变体；不声称旧target指标未知。','same-grid best仅作诊断，不参与正式判定。单seed结果不能证明跨seed收益或relation pollution的因果机制。','未追加训练。'+('S3未通过，按用户要求停止spectral gate/token变体，不调门槛或配置。' if not go else 'S3通过筛选，但未启动后续三seed×200轮。')]
 (s.OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n');(s.OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n\nLOCKED PLAN\n'+json.dumps(s.PLAN,indent=2)+'\n\nFULL SUMMARY\n'+json.dumps(summary,indent=2)+'\n\nPER CLASS FIXED100\n'+(s.OUT/'per_class_fixed100_deltas.csv').read_text());s.verify();s.dump(s.OUT/'FINAL_REVIEW.json',{'preflight_passed':True,'all100_epochs_paired_data_RNG_BN_calls_passed':True,'fixed100_only_Go':True,'no_class_excluded_from_damage_check':True,'no_new_training_other_than_S3':True,'no_followup_started':True,'decision':summary['decision']});print('S3_REPORT_COMPLETE',summary['decision'],json.dumps(fixed),json.dumps(recovery),flush=True)
if __name__=='__main__':main()
