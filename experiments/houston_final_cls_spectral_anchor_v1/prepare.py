import json
import study as s
assert not (s.OUT/'code_manifest.json').exists()
s.dump(s.OUT/'PLAN.json',s.PLAN)
previous=json.loads((s.previous()/'FINAL_DELIVERY.json').read_text())['files_sha256'];paths=[s.previous()/'S0_2100/frozen_manifest.json',s.previous()/'S0_2100/history.jsonl',s.previous()/'S0_2100/config.json',s.previous()/'S0_2100/offline_predictions_frozen.json']+[s.previous()/'S0_2100/offline_predictions'/('epoch_%03d.npz'%e) for e in s.PLAN['grid']]
for p in paths:assert s.sha(p)==previous[str(p.relative_to(s.previous()))]
paths += [s.ROOT/'experiments/houston_persistent_spectral_anchor_v1/anchor.py',s.baseline()/'config.json',s.baseline()/'history.jsonl']
s.dump(s.OUT/'baseline_manifest.json',{str(p):s.sha(p) for p in paths})
s.dump(s.OUT/'code_manifest.json',{str(p):s.sha(p) for p in sorted(s.CODE.glob('*.py'))})
s.verify();print('S3_PLAN_LOCKED')
