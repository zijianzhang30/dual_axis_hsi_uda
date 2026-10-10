import json
import study as s
assert not (s.OUT/'code_manifest.json').exists()
old=json.loads((s.previous()/'FINAL_DELIVERY.json').read_text())['files_sha256'];paths=[s.relation()/x for x in ('config.json','history.jsonl','paired_rng.pt','frozen_manifest.json','offline_predictions_frozen.json','checkpoints/epoch_100.pth','offline_predictions/epoch_100.npz')]
for p in paths:assert s.sha(p)==old[str(p.relative_to(s.previous()))]
s.dump(s.OUT/'PLAN.json',s.PLAN);s.dump(s.OUT/'baseline_manifest.json',{str(p):s.sha(p) for p in paths});s.dump(s.OUT/'code_manifest.json',{str(p):s.sha(p) for p in sorted(s.CODE.glob('*.py'))});s.verify();print('DUAL_EVIDENCE_PLAN_LOCKED')
