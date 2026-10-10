import csv,json,numpy as np
import study as s

def main():
 s.verify();allrecords={}
 for arm in s.ARMS:
  folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==100 and mf['data_RNG_all_epochs_match_original_C'];q=json.loads((folder/'offline_predictions_frozen.json').read_text());assert q['target_GT_not_scored'] and q['model_and_running_stats_unchanged'];allrecords[arm]=q['records']
  for r in q['records']:assert s.sha(r['path'])==r['sha256']
 s.dump(s.OUT/'ALL_PREDICTIONS_FROZEN_BEFORE_GT.json',{'all_three_arms_ten_checkpoints_hashed':True,'records':allrecords,'fixed100_Go_formula':s.PLAN['Go'],'target_GT_scoring_begins_after_this_barrier':True})
 cache=s.B/'tokenizer_stem_value_diagnostic_v1';prov=json.loads((cache/'provenance.json').read_text());yp=cache/'data/target_labels.npy';coords=cache/'data/target_coords.npy';assert s.sha(yp)==prov['data_sha256']['target_labels.npy'] and s.sha(coords)==prov['data_sha256']['target_coords.npy'];y=np.load(yp);xy=np.load(coords);helper=s.module('anchor_metrics',s.B/'tokenizer_mechanism_v2/snapshot/helpers.py');rows=[];pc=[];grid=[]
 for arm,rs in allrecords.items():
  ev=[]
  for r in rs:
   z=np.load(r['path']);assert np.array_equal(z['coordinates'],xy);v=helper.metrics(y,z['predictions'],z['logits'],7);ev.append({'epoch':r['epoch'],'metrics':v});grid.append({'arm':arm,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa']})
  s.dump(s.OUT/(arm+'_scored_grid.json'),ev)
  for endpoint,r in [('fixed100',ev[-1]),('same_grid_best_diagnostic',max(ev,key=lambda e:e['metrics']['oa_percent']))]:
   v=r['metrics'];rows.append({'arm':arm,'seed':2100,'endpoint':endpoint,'epoch':r['epoch'],'OA':v['oa_percent'],'AA':v['aa_percent'],'Kappa':v['kappa'],'zero_recall_classes':[i+1 for i,x in enumerate(v['per_class_percent']) if x==0],'collapse95':max(v['prediction_shares'])>=.95,'max_prediction_share':max(v['prediction_shares'])});pc.extend([{'arm':arm,'endpoint':endpoint,'epoch':r['epoch'],'class':i+1,'recall':x,'prediction_share':v['prediction_shares'][i]} for i,x in enumerate(v['per_class_percent'])])
 delta=[]
 for endpoint in ('fixed100','same_grid_best_diagnostic'):
  rr={r['arm']:r for r in rows if r['endpoint']==endpoint};base=rr['S0']
  for arm in ('S1','S2'):delta.append({'arm':arm,'endpoint':endpoint,**{k+'_delta':rr[arm][k]-base[k] for k in ('OA','AA','Kappa')},'candidate_epoch':rr[arm]['epoch'],'baseline_epoch':base['epoch']})
 fixed={r['arm']:r for r in rows if r['endpoint']=='fixed100'};d=next(r for r in delta if r['arm']=='S2' and r['endpoint']=='fixed100');gate={'OA_delta_at_least_0_5pp':d['OA_delta']>=.5,'AA_delta_at_least_minus_0_5pp':d['AA_delta']>=-.5,'no_new_zero_recall':not set(fixed['S2']['zero_recall_classes'])-set(fixed['S0']['zero_recall_classes']),'no_new_collapse':not fixed['S2']['collapse95'] or fixed['S0']['collapse95']};go=all(gate.values());summary={'fixed_and_diagnostic_endpoints':rows,'deltas':delta,'Go_conditions':gate,'decision':'Go' if go else 'No-Go','followups_started':False,'three_seed_200_requires_separate_authorization':go,'single_seed_screening_not_cross_seed_evidence':True,'preflight':json.loads((s.OUT/'preflight.json').read_text())};s.dump(s.OUT/'summary.json',summary)
 for name,rr in [('endpoints.csv',rows),('paired_deltas.csv',delta),('per_class_recall.csv',pc),('same_grid_diagnostic.csv',grid)]:
  with (s.OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 lines=['houston_persistent_spectral_anchor_v1','seed2100 / source CE only / fixed100','Decision: '+summary['decision'],'','arm     OA(%)    AA(%)    Kappa']
 for arm in s.ARMS:r=fixed[arm];lines.append('%s  %.4f  %.4f  %.6f'%(arm,r['OA'],r['AA'],r['Kappa']))
 for r in delta:
  if r['endpoint']=='fixed100':lines.append('%s-S0: OA %+.4fpp; AA %+.4fpp; Kappa %+.6f'%(r['arm'],r['OA_delta'],r['AA_delta'],r['Kappa_delta']))
 lines+=['','Locked Go conditions: '+json.dumps(gate),'Zero recall: '+json.dumps({a:r['zero_recall_classes'] for a,r in fixed.items()}),'Collapse95: '+json.dumps({a:r['collapse95'] for a,r in fixed.items()}),'','S2 alpha=0: all train/eval logits, original gradients, first SGD base update, CPU/CUDA RNG and BN buffers/counts bitwise equal S0. Alpha receives gradient immediately; spectral encoder receives gradient after alpha becomes nonzero.','S0 fresh epoch100 equals historical original C checkpoint bitwise. All100 epoch data/augmentation/global RNG hashes and original forward/BN update counts matched original C for all3 arms.','S1 sequence length6 changes dropout masks; pre/post forward RNG replay preserves original global/data RNG stream. S2 native stream unchanged.','All30 checkpoint predictions hashed before target GT scoring. Target labels are zero in training/inference loaders; native candidate mask remains original protocol.','Same-grid best is target-oracle diagnostic only, not used in the Go decision. No additional variants or trainings launched.']
 (s.OUT/'FINDINGS.txt').write_text('\n'.join(lines)+'\n');(s.OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n\nLOCKED PROTOCOL\n'+json.dumps(s.PLAN,indent=2)+'\n\nFULL SUMMARY\n'+json.dumps(summary,indent=2)+'\n\nPER CLASS RECALL\n'+(s.OUT/'per_class_recall.csv').read_text());s.verify();s.dump(s.OUT/'FINAL_REVIEW.json',{'all_authorized_three_runs_finished':True,'all_target_predictions_frozen_before_GT':True,'locked_gate_applied_without_tuning':True,'fixed100_only_formal':True,'preflight_passed':True,'followup_started':False,'decision':summary['decision']});print('COMPLETE',summary['decision'],json.dumps(fixed),flush=True)
if __name__=='__main__':main()
