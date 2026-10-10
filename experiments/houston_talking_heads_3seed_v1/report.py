import csv,json
import numpy as np
import study as s

def dump_csv(name,rows):
 with (s.OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 s.verify();records={};audit={};compute=[];heads=[]
 for seed in s.SEEDS:
  ref=[json.loads(x) for x in (s.baseline(seed)/'history.jsonl').read_text().splitlines()]
  for arm in ('TH0',)+s.ARMS:
   p=s.prediction_manifest(arm,seed);mf=json.loads(p.read_text());rr=mf['records'];assert [r['epoch'] for r in rr]==s.PLAN['grid'];assert all(s.sha(r['path'])==r['sha256'] for r in rr);records[arm,seed]=rr
   if arm=='TH0':continue
   f=p.parent;cfg=json.loads((f/'config.json').read_text());h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];fm=json.loads((f/'frozen_manifest.json').read_text());assert fm['completed_epochs']==200 and fm['optimizer_steps']==3600 and len(h)==200
   assert cfg['seed']==seed and cfg['base_audit']['indices_sha256']==json.loads((s.baseline(seed)/'config.json').read_text())['indices_sha256']
   assert cfg['plan']['epochs']==200 and cfg['plan']['grid']==s.PLAN['grid'] and cfg['plan']['loss']==s.PLAN['loss']
   assert all(x['batch_stream_sha256']==y['batch_stream_sha256'] and x['original_rng_stream_sha256']==y['original_rng_stream_sha256'] for x,y in zip(h,ref))
   assert all(x['steps']==18 and all(v==18*x['epoch'] for v in x['BN_counts'].values()) and x['normal_forward_calls']=={'block0':18,'block1':18,'block2':18,'classifier':72} and not x['target_GT_accessed'] for x in h)
   assert all(s.sha(r['path'])==r['sha256'] for r in fm['checkpoints'])
   audit['%s_%d'%(arm,seed)]={'all200_data_main_RNG_match_TH0':True,'optimizer_steps':3600,'BN_and_Transformer_calls_unchanged':True,'source_indices_match':True,'all_checkpoint_hashes_valid':True}
   compute.append({'arm':arm,'seed':seed,'parameters':cfg['parameters'],'added_parameters':cfg['install_audit']['added_parameters'],'train_seconds':fm['train_seconds'],'evaluation_seconds':mf['seconds'],'source_val_final':fm['source_val_final']})
   for row in h:
    d=row['first_batch_diagnostics']['attention'];g=row['first_batch_diagnostics']['new_parameter_gradients'];heads.append({'arm':arm,'seed':seed,'epoch':row['epoch'],'pre_offdiag_norm':d['talk_pre']['offdiagonal_Frobenius'],'post_offdiag_norm':d['talk_post']['offdiagonal_Frobenius'],'pre_grad_norm':g['talk_pre']['norm'],'post_grad_norm':g['talk_post']['norm'],'pre_offdiag_grad_norm':g['talk_pre']['offdiagonal_norm'],'post_offdiag_grad_norm':g['talk_post']['offdiagonal_norm'],'negative_postmix_fraction':d['postmix_negative_weight_fraction'],'entropy_before_postmix':d['head_softmax_entropy_BEFORE_postmix'],'postmix_row_mass_mean':d['head_postmix_row_mass_mean']})
 s.dump(s.OUT/'OFFLINE_SCORING_BARRIER.json',{'all_predictions_frozen_before_new_GT_scoring':True,'reused_results_previously_GT_evaluated':True,'no_target_GT_parameter_selection':True});s.dump(s.OUT/'final_pairing_audit.json',{'passed':True,'runs':audit})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());lp=cache/'data/target_labels.npy';assert s.sha(lp)==prov['data_sha256']['target_labels.npy'];y=np.load(lp);helper=s.module('talking_3seed_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');ends=[];cls=[];counts=np.bincount(y,minlength=7).tolist()
 for (arm,seed),rr in records.items():
  grid=[]
  for r in rr:
   f=np.load(r['path']);grid.append({'epoch':r['epoch'],'metrics':helper.metrics(y,f['predictions'],f['logits'],7),'attention_diagnostics':r.get('first128_eval_attention_diagnostics')})
  s.dump(s.OUT/('%s_%d_scored_grid.json'%(arm,seed)),grid)
  for ep,r in [('fixed200',grid[-1]),('target_best',max(grid,key=lambda r:r['metrics']['oa_percent']))]:
   m=r['metrics'];ends.append({'arm':arm,'seed':seed,'endpoint':ep,'epoch':r['epoch'],'OA':m['oa_percent'],'AA':m['aa_percent'],'Kappa':m['kappa'],'collapse95':max(m['prediction_shares'])>=.95,'zero_recall_classes':[i+1 for i,v in enumerate(m['per_class_percent']) if v==0]});cls.extend([{'arm':arm,'seed':seed,'endpoint':ep,'class':i+1,'samples':counts[i],'recall':v,'prediction_share':m['prediction_shares'][i]} for i,v in enumerate(m['per_class_percent'])])
 paired=[];aggregates=[];paired_ag=[]
 for ep in ('target_best','fixed200'):
  for seed in s.SEEDS:
   r={x['arm']:x for x in ends if x['endpoint']==ep and x['seed']==seed}
   for a,b in [('TH2','TH0'),('TH2','TH1'),('TH1','TH0')]:paired.append({'endpoint':ep,'seed':seed,'candidate':a,'baseline':b,**{k+'_delta':r[a][k]-r[b][k] for k in ('OA','AA','Kappa')}})
  for arm in ('TH0',)+s.ARMS:
   r=[x for x in ends if x['endpoint']==ep and x['arm']==arm];aggregates.append({'endpoint':ep,'arm':arm,**{k+q:float(np.mean([x[k] for x in r])) if q=='_mean' else float(np.std([x[k] for x in r],ddof=1)) for k in ('OA','AA','Kappa') for q in ('_mean','_sample_std')}})
  for a,b in [('TH2','TH0'),('TH2','TH1'),('TH1','TH0')]:
   r=[x for x in paired if x['endpoint']==ep and x['candidate']==a and x['baseline']==b];paired_ag.append({'endpoint':ep,'candidate':a,'baseline':b,**{k+q:float(np.mean([x[k+'_delta'] for x in r])) if q=='_mean_delta' else float(np.std([x[k+'_delta'] for x in r],ddof=1)) for k in ('OA','AA','Kappa') for q in ('_mean_delta','_paired_sample_std')},'OA_positive_seeds':sum(x['OA_delta']>0 for x in r),'AA_positive_seeds':sum(x['AA_delta']>0 for x in r),'OA_AA_both_positive_seeds':sum(x['OA_delta']>0 and x['AA_delta']>0 for x in r),'worst_OA_delta':min(x['OA_delta'] for x in r),'worst_AA_delta':min(x['AA_delta'] for x in r)})
 summary={'endpoints':ends,'aggregates':aggregates,'paired':paired,'paired_aggregates':paired_ag,'compute':compute,'native_class_counts':counts,'target_best_uses_target_GT_oracle_selection':True,'AA_Kappa_from_same_OA_best_checkpoint':True,'same_20checkpoint_grid':True,'sample_standard_deviation_ddof':1,'seed2100_reused_not_independent_new_replication':True,'new_training_count':4,'attention_code_identical_to_seed2100':True,'TH1_TH2_not_parameter_matched':True,'TH2_extra_offdiagonal_parameters_over_TH1':112,'existing_TalkingHeads_not_established_original_contribution':True,'no_auto_followup':True};s.dump(s.OUT/'summary.json',summary)
 for n,r in [('endpoints.csv',ends),('aggregate_metrics.csv',aggregates),('paired_deltas.csv',paired),('paired_aggregate.csv',paired_ag),('per_class_recall.csv',cls),('head_diagnostics.csv',heads)]:dump_csv(n,r)
 lines=['Houston TalkingHeads firstselflayer3seed repeatabilityvalidation;seeds2100/2101/2102;200epochs;sourceCEonly.','ReuseallTH0andTH1/TH2seed2100;fournewtrainings;noarchitecture/loss/protocolchanges.','Target-best usesidentical10:10:200grid,AA/Kappa fromsameOA-bestcheckpoint;targetGToracle selection. Fixed200separate. Sample std ddof1.']
 for r in aggregates:lines.append('%s %s OA %.6f +/- %.6f AA %.6f +/- %.6f Kappa %.6f +/- %.6f'%(r['endpoint'],r['arm'],r['OA_mean'],r['OA_sample_std'],r['AA_mean'],r['AA_sample_std'],r['Kappa_mean'],r['Kappa_sample_std']))
 for r in paired_ag:lines.append('%s %s-%s OA %+.6f AA %+.6f Kappa %+.6f bothpositive %d/3 worstOA %+.6f worstAA %+.6f'%(r['endpoint'],r['candidate'],r['baseline'],r['OA_mean_delta'],r['AA_mean_delta'],r['Kappa_mean_delta'],r['OA_AA_both_positive_seeds'],r['worst_OA_delta'],r['worst_AA_delta']))
 for r in ends:lines.append('%s seed%d %s epoch%d OA%.6f AA%.6f Kappa%.6f zero%s collapse%s'%(r['arm'],r['seed'],r['endpoint'],r['epoch'],r['OA'],r['AA'],r['Kappa'],r['zero_recall_classes'],r['collapse95']))
 lines+=['No bestseed cherry-picking,independent AA/Kappa maxima,or automatic extra variants/training.','TalkingHeads is anexisting mechanism;repeatable gains would showcomponent effectiveness,not establish originality. TH1adds16,TH2adds128parameters (112extraoffdiagonals);notcapacitymatched.','GPUtimings underdifferent sharedloads are not purecomputeoverhead estimates. Seeperclassrecall andpairedseeddeltas beforeclaimingclasscoverage stability.']
 (s.OUT/'CONCLUSION.txt').write_text('\n'.join(lines)+'\n');s.verify();s.dump(s.OUT/'status.json',{'status':'completed_3seed_validation','auto_followup':False});print('TH_3SEED_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
