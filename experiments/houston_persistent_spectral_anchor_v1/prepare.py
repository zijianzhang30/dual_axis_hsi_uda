import study as s
s.OUT.mkdir(exist_ok=True);assert not (s.OUT/'code_manifest.json').exists()
s.dump(s.OUT/'PLAN.json',s.PLAN)
s.dump(s.OUT/'code_manifest.json',{str(p):s.sha(p) for p in sorted(s.CODE.glob('*.py'))})
s.dump(s.OUT/'baseline_manifest.json',{str(p):s.sha(p) for p in [s.baseline()/'config.json',s.baseline()/'history.jsonl',s.baseline()/'checkpoints/epoch_100.pth']})
s.verify();print('PLAN_LOCKED',flush=True)
