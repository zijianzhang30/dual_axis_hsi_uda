import json
import study as s
assert not (s.OUT/'code_manifest.json').exists();paths={};audits=[]
for d in s.DATASETS:
 for seed in s.SEEDS:
  cf=s.reference(d,seed);ch=[json.loads(x) for x in (cf/'history.jsonl').read_text().splitlines()];assert len(ch)==200
  for r in ('J_A','J_C'):
   f=s.reused_relation(d,r,seed)
   if f is None:continue
   hist=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(hist)==200
   if r=='J_A':assert all(a['batch_stream_sha256']==b['batch_stream_sha256'] and a['original_rng_stream_sha256']==b['original_rng_stream_sha256'] for a,b in zip(hist,ch))
   audits.append({'dataset':d,'relation':r,'seed':seed,'path':str(f),'200epoch_data_RNG_match':True})
   for p in [f/'config.json',f/'history.jsonl',f/'frozen_manifest.json']+list((f/'checkpoints').glob('epoch_*.pth')):paths[str(p)]=s.sha(p)
  for epoch in range(10,201,10):
   for scope in (('native','full') if d=='hyrank' else ('native',)):
    p=s.spec_file(d,seed,epoch,scope);paths[p['path']]=p['sha256']
  for p in (cf/'history.jsonl',cf/'actual_batches.jsonl'):
   if p.exists():paths[str(p)]=s.sha(p)
for p in (s.ROOT/'experiments/cross_benchmark_relation_identity_dual_evidence_v1').glob('*.py'):paths[str(p)]=s.sha(p)
for p in (s.ROOT/'experiments/houston_relation_identity_dual_evidence_3seed_v1').glob('*.py'):paths[str(p)]=s.sha(p)
# Add frozen runtime protocol files to provenance without modifying prior studies.
for name in ('tokenizer_mechanism_v2','tokenizer_mechanism_pavia_l2_fixed_v1','hyrank_joint_lmmd_eq9_v1','hyrank_ce_eq9_off_2100_v1','hyrank_lmmd_eq9_off_2100_v1'):
 f=s.NAS/name
 for p in list(f.glob('*.py'))+list((f/'snapshot').rglob('*.py'))+list((f/'runtime_snapshot').rglob('*.py'))+list((f/'experiment_code').glob('*.py')):paths[str(p)]=s.sha(p)
assert len(s.new_tasks())==21;s.dump(s.OUT/'PLAN.json',s.PLAN);s.dump(s.OUT/'REUSE_AUDIT.json',audits);s.dump(s.OUT/'reuse_manifest.json',paths);s.dump(s.OUT/'code_manifest.json',{str(p):s.sha(p) for p in s.CODE.glob('*.py')});s.verify();print('FACT0RIAL_PLAN_LOCKED:21 new relation runs,72 derived groups,all9 spectral branches reused')
