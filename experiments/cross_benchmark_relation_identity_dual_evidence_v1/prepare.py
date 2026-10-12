import json
import study as s
assert not (s.OUT/'code_manifest.json').exists();paths=[]
for dataset in s.DATASETS:
 for seed in s.SEEDS:
  f=s.baseline(dataset,seed);paths += [f/'config.json',f/'history.jsonl',f/'frozen_manifest.json']
  for r in s.baseline_records(dataset,seed):
   assert s.sha(r['checkpoint'])==r['checkpoint_sha256'];paths.append(__import__('pathlib').Path(r['checkpoint']))
   if 'prediction' in r:
    for name in ('prediction','full_prediction'):assert s.sha(r[name])==r[name+'_sha256'];paths.append(__import__('pathlib').Path(r[name]))
  if dataset=='pavia':paths.append(f/'predictions_200.npz')
  else:paths.append(f/'actual_batches.jsonl')
# Include all frozen protocol code dependencies and Houston fixed design.
for folder in ('tokenizer_mechanism_pavia_l2_fixed_v1','hyrank_joint_lmmd_eq9_v1','hyrank_ce_eq9_off_2100_v1'):
 for name in ('provenance.json','code_manifest.json','original_files_manifest.json'):
  p=s.B/folder/name
  if p.exists():paths.append(p)
paths += list((s.B/'tokenizer_mechanism_pavia_l2_fixed_v1').glob('*.py'));paths += list((s.B/'tokenizer_mechanism_pavia_l2_fixed_v1/snapshot').rglob('*.py'));paths += list((s.B/'hyrank_joint_lmmd_eq9_v1').glob('*.py'));paths += list((s.B/'hyrank_joint_lmmd_eq9_v1/runtime_snapshot').rglob('*.py'));paths += [s.ROOT/'experiments/houston_relation_identity_dual_evidence_3seed_v1/expert.py',s.ROOT/'results/houston_relation_identity_dual_evidence_3seed_v1/PLAN.json']
s.dump(s.OUT/'PLAN.json',s.PLAN);s.dump(s.OUT/'baseline_manifest.json',{str(p):s.sha(p) for p in paths});s.dump(s.OUT/'code_manifest.json',{str(p):s.sha(p) for p in sorted(s.CODE.glob('*.py'))});s.verify();print('CROSS_BENCHMARK_PLAN_LOCKED')
