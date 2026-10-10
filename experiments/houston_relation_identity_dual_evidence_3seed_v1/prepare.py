import study as s
assert not (s.OUT/'code_manifest.json').exists()
paths=[]
for seed in s.SEEDS:
 paths += [s.baseline(seed)/n for n in ('config.json','history.jsonl','predictions_200.npz')]
 paths += [s.baseline(seed)/'checkpoints'/('epoch_%03d.pth'%e) for e in s.PLAN['grid']]
paths += [s.ROOT/'experiments/houston_relation_identity_dual_evidence_v1/expert.py',s.ROOT/'results/houston_relation_identity_dual_evidence_v1/spectral_epoch_100.pth']
assert s.sha(s.CODE/'expert.py')==s.sha(paths[-2])
s.dump(s.OUT/'PLAN.json',s.PLAN);s.dump(s.OUT/'baseline_manifest.json',{str(p):s.sha(p) for p in paths});s.dump(s.OUT/'code_manifest.json',{str(p):s.sha(p) for p in sorted(s.CODE.glob('*.py'))});s.verify();print('PLAN_LOCKED')
