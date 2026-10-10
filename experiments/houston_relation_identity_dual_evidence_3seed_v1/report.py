import csv,json,numpy as np
import study as s

def write(name,rows):
 with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 s.verify();files={}
 for seed in s.SEEDS:
  f=s.OUT/('seed_%d'%seed);h=json.loads((f/'TRAINING_COMPLETE_FROZEN.json').read_text());q=json.loads((f/'ALL_GRID_PREDICTIONS_FROZEN.json').read_text());assert h['epochs']==200 and h['all200_data_RNG_BN_and_calls_match_original_C'] and h['all20_relation_checkpoints_bitwise_match_original_C'] and q['no_target_GT_scored'];files[seed]=q['records'];assert [r['epoch'] for r in files[seed]]==s.PLAN['grid']
  for r in files[seed]:assert s.sha(r['path'])==r['sha256']
 s.dump(s.OUT/'ALL3_SEEDS_ALL_GRID_FROZEN_BEFORE_GT.json',{'all3_training_complete':True,'records':files,'all60_grid_files_four_arms_and_disagreement_hashed':True,'target_GT_scoring_starts_after_this_barrier':True,'prior_Houston_target_metrics_known':True,'fixed200_only_formal':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());yp=cache/'data/target_labels.npy';cp=cache/'data/target_coords.npy';assert s.sha(yp)==prov['data_sha256']['target_labels.npy'] and s.sha(cp)==prov['data_sha256']['target_coords.npy'];y=np.load(yp);coords=np.load(cp);helper=s.module('dual3_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];pc=[];grids=[];rec=[];subset=[];both=[];classrec=[]
 for seed,rs in files.items():
  scored={k:[] for k in ('R0','R1','R2','R3')};fixed=None
  for r in rs:
   z=np.load(r['path']);assert np.array_equal(z['coordinates'],coords)
   for k in scored:
    v=helper.metrics(y,z[k+'_predictions'],z[k+'_logits'],7);scored[k].append({'epoch':r['epoch'],'metrics':v});grids.append({'seed':seed,'arm':k,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa']})
   if r['epoch']==200:fixed={k:z[k+'_predictions'].copy() for k in scored};assert np.array_equal(z['agreement'],fixed['R0']==fixed['R1']) and np.array_equal(z['disagreement'],fixed['R0']!=fixed['R1'])
  s.dump(s.OUT/('seed_%d_scored_grid.json'%seed),scored)
  for k,ev in scored.items():
   for endpoint,r in [('fixed200',ev[-1]),('same_grid_target_best_diagnostic',max(ev,key=lambda q:q['metrics']['oa_percent']))]:
    v=r['metrics'];ends.append({'seed':seed,'arm':k,'endpoint':endpoint,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa'],'zero_recall_classes':[i+1 for i,x in enumerate(v['per_class_percent']) if x==0],'collapse95':max(v['prediction_shares'])>=.95,'max_prediction_share':max(v['prediction_shares'])});pc.extend([{'seed':seed,'arm':k,'endpoint':endpoint,'epoch':r['epoch'],'class':i+1,'n':int((y==i).sum()),'recall':x,'prediction_share':v['prediction_shares'][i]} for i,x in enumerate(v['per_class_percent'])])
  old=fixed['R0']==y;spec=fixed['R1']==y;agree=fixed['R0']==fixed['R1'];bw=(~old)&(~spec);assert np.array_equal(bw&~agree,(~agree)&(~old)&(~spec))
  for k in ('R2','R3'):
   now=fixed[k]==y;recovered=int((~old&now).sum());damaged=int((old&~now).sum());assert recovered-damaged==int(now.sum())-int(old.sum());rec.append({'seed':seed,'arm':k,'R0_errors':int((~old).sum()),'recovered':recovered,'recovery_percent':float(recovered/(~old).sum()*100),'R0_correct':int(old.sum()),'damaged':damaged,'damage_percent':float(damaged/old.sum()*100)})
   for c in range(7):
    mask=y==c;bad=mask&~old;good=mask&old;classrec.append({'seed':seed,'arm':k,'class':c+1,'R0_errors':int(bad.sum()),'recovered':int((bad&now).sum()),'R0_correct':int(good.sum()),'damaged':int((good&~now).sum())})
  recovered=bw&(fixed['R3']==y);both.append({'seed':seed,'both_top1_wrong_all':int(bw.sum()),'both_top1_wrong_disagreement':int((bw&~agree).sum()),'R3_recovered':int(recovered.sum()),'percent_of_all_both_wrong':float(recovered.sum()/bw.sum()*100),'percent_of_disagreement_both_wrong':float(recovered.sum()/(bw&~agree).sum()*100),'percent_of_R0_errors':float(recovered.sum()/(~old).sum()*100),'percent_of_R3_recoveries':float(recovered.sum()/(~old&(fixed['R3']==y)).sum()*100) if (~old&(fixed['R3']==y)).any() else float('nan')})
  for group,mask in [('all',np.ones(len(y),bool)),('agreement',agree),('disagreement',~agree)]:
   for c in [None]+list(range(7)):
    use=mask if c is None else mask&(y==c);n=int(use.sum());subset.append({'seed':seed,'subset':group,'class':'all' if c is None else c+1,'n':n,**{k+'_accuracy':float((fixed[k][use]==y[use]).mean()*100) if n else float('nan') for k in ('R0','R1','R2','R3')},'any_expert_top1_accuracy':float((old|spec)[use].mean()*100) if n else float('nan')})
 deltas=[]
 for endpoint in ('fixed200','same_grid_target_best_diagnostic'):
  for seed in s.SEEDS:
   rr={r['arm']:r for r in ends if r['endpoint']==endpoint and r['seed']==seed}
   for k in ('R1','R2','R3'):deltas.append({'seed':seed,'arm':k,'endpoint':endpoint,'candidate_epoch':rr[k]['epoch'],'baseline_epoch':rr['R0']['epoch'],**{m+'_delta':rr[k][m]-rr['R0'][m] for m in ('OA','AA','Kappa')}})
 agg=[]
 for endpoint in ('fixed200','same_grid_target_best_diagnostic'):
  for k in ('R0','R1','R2','R3'):
   rr=[r for r in ends if r['endpoint']==endpoint and r['arm']==k];agg.append({'arm':k,'endpoint':endpoint,**{m+'_mean':float(np.mean([r[m] for r in rr])) for m in ('OA','AA','Kappa')},**{m+'_std':float(np.std([r[m] for r in rr],ddof=1)) for m in ('OA','AA','Kappa')}})
 deltaagg=[]
 for endpoint in ('fixed200','same_grid_target_best_diagnostic'):
  for k in ('R1','R2','R3'):
   rr=[r for r in deltas if r['endpoint']==endpoint and r['arm']==k];deltaagg.append({'arm':k,'endpoint':endpoint,**{m+'_delta_mean':float(np.mean([r[m+'_delta'] for r in rr])) for m in ('OA','AA','Kappa')},**{m+'_delta_std':float(np.std([r[m+'_delta'] for r in rr],ddof=1)) for m in ('OA','AA','Kappa')}})
 pa=[]
 for k in ('R0','R1','R2','R3'):
  for c in range(1,8):
   rr=[r['recall'] for r in pc if r['endpoint']=='fixed200' and r['arm']==k and r['class']==c];dd=[next(r['recall'] for r in pc if r['seed']==seed and r['endpoint']=='fixed200' and r['arm']==k and r['class']==c)-next(r['recall'] for r in pc if r['seed']==seed and r['endpoint']=='fixed200' and r['arm']=='R0' and r['class']==c) for seed in s.SEEDS];pa.append({'arm':k,'class':c,'recall_mean':float(np.mean(rr)),'recall_std':float(np.std(rr,ddof=1)),'delta_vs_R0_mean':float(np.mean(dd)),'delta_vs_R0_std':float(np.std(dd,ddof=1))})
 rr=[r for r in deltas if r['endpoint']=='fixed200' and r['arm']=='R3'];r3=[r for r in ends if r['endpoint']=='fixed200' and r['arm']=='R3'];gate={'mean_deltaOA_positive':np.mean([r['OA_delta'] for r in rr])>0,'at_least_2_of_3_positive_deltaOA':sum(r['OA_delta']>0 for r in rr)>=2,'mean_deltaAA_nonnegative':np.mean([r['AA_delta'] for r in rr])>=0,'at_least_2_of_3_nonnegative_deltaAA':sum(r['AA_delta']>=0 for r in rr)>=2,'no_R3_collapse_any_seed':not any(r['collapse95'] for r in r3),'no_R3_zero_recall_any_seed':not any(r['zero_recall_classes'] for r in r3)};gate={k:bool(v) for k,v in gate.items()};go=all(gate.values())
 def ms(rows,keys):return {k:{'mean':float(np.mean([r[k] for r in rows])),'sample_std':float(np.std([r[k] for r in rows],ddof=1))} for k in keys}
 rates=ms([r for r in rec if r['arm']=='R3'],('recovery_percent','damage_percent'));ds=ms([r for r in subset if r['subset']=='disagreement' and r['class']=='all'],('R0_accuracy','R1_accuracy','R3_accuracy'));bwagg=ms(both,('percent_of_all_both_wrong','percent_of_disagreement_both_wrong','percent_of_R0_errors','percent_of_R3_recoveries'))
 summary={'decision':'Go' if go else 'No-Go','conditions':gate,'aggregate':agg,'paired_delta_aggregate':deltaagg,'same_seed_deltas':deltas,'endpoints':ends,'per_class_fixed200_aggregate':pa,'R3_recovery_damage_mean_std':rates,'disagreement_accuracy_mean_std':ds,'both_top1_wrong_recovery_mean_std':bwagg,'all_experts_independently_source_CE_trained_200':True,'all20_relation_checkpoints_and_original200_logits_exact':True,'target_GT_only_after_all3_freeze':True,'best_diagnostic_not_used_for_gate':True,'no_new_fusion_or_cross_dataset_started':True,'sample_std_ddof':1};s.dump(s.OUT/'summary.json',summary)
 for name,rows in [('fixed_and_best_endpoints.csv',ends),('mean_std_metrics.csv',agg),('paired_deltas.csv',deltas),('paired_delta_mean_std.csv',deltaagg),('per_class_recall.csv',pc),('per_class_mean_std.csv',pa),('recovery_damage.csv',rec),('per_class_recovery_damage.csv',classrec),('agreement_disagreement_accuracy.csv',subset),('both_experts_wrong_recovery.csv',both),('same_grid_diagnostic.csv',grids)]:write(name,rows)
 lines=['houston_relation_identity_dual_evidence_3seed_v1','判定：'+summary['decision'],'三个seed2100/2101/2102均重新独立训练relation及spectral专家200轮，仅source CE；架构、SGD、温度、融合权重完全沿用前版。','原C每10轮checkpoint与历史原C逐位一致；全部200轮数据/RNG/BN/调用计数匹配；seed2100光谱epoch100与前版逐位一致。','所有3seed训练及20点网格四方案预测/坐标/agreement均先冻结并hash，再读target GT评分。','','fixed200 mean ± sample std (ddof=1)']
 for r in agg:
  if r['endpoint']=='fixed200':lines.append('%s OA %.4f±%.4f AA %.4f±%.4f Kappa %.6f±%.6f'%(r['arm'],r['OA_mean'],r['OA_std'],r['AA_mean'],r['AA_std'],r['Kappa_mean'],r['Kappa_std']))
 lines+=['','同seed fixed200 R3-R0']
 for r in rr:lines.append('seed%d: deltaOA %+.4fpp deltaAA %+.4fpp deltaKappa %+.6f'%(r['seed'],r['OA_delta'],r['AA_delta'],r['Kappa_delta']))
 lines+=['锁定Go条件：'+json.dumps(gate),'R3原错误恢复/原正确破坏：'+json.dumps(rates),'disagreement各分支准确率：'+json.dumps(ds),'两expert top1均错但R3恢复：'+json.dumps(bwagg),'','AA“不系统性下降”预先定义为平均deltaAA≥0且至少2/3seed deltaAA≥0；不根据GT调整该解释。','both-top1-wrong恢复比例同时给出全部both-wrong、disagreement both-wrong、R0全部错误及R3恢复的分母。','任一expert top1正确只是二选一selector上界，不是全类别概率融合上界；R3可预测两个expert都未取top1的正确类。','same-grid target-best仅oracle诊断，各组best可在不同epoch；AA/Kappa取同OA-best epoch，不参与Go。','无target pseudo-label/GT训练、LMMD、gate、温度或权重搜索；此前Houston结果已知，不宣称未触及的独立测试。','未自动追加变体或Pavia/HyRANK训练。'+('通过本轮稳定性门槛，跨数据集验证与正式方法设计须另行确定。' if go else '未通过预定门槛，停止进一步调整融合方式。')]
 (s.OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n');(s.OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n\nLOCKED PLAN\n'+json.dumps(s.PLAN,indent=2)+'\n\nFULL SUMMARY\n'+json.dumps(summary,indent=2)+'\n\nPER CLASS FIXED200\n'+(s.OUT/'per_class_mean_std.csv').read_text());s.verify();s.dump(s.OUT/'FINAL_REVIEW.json',{'all3_seed_200_training_complete':True,'all60_grid_files_frozen_before_GT':True,'original_C_relation_all_checkpoints_bitwise_exact':True,'spectral2100_first100_bitwise_replayed':True,'fixed200_only_formal':True,'sample_std_ddof1':True,'decision':summary['decision'],'no_followup_started':True});print('REPORT_COMPLETE',summary['decision'],json.dumps([r for r in agg if r['endpoint']=='fixed200']),flush=True)
if __name__=='__main__':main()
