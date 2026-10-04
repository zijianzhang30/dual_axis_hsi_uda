"""Read-only collection after all four fixed500 evaluations exist."""
import json
from pathlib import Path

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_joint_adapt_2x2_v1')
CE = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1')
paths = dict(B=CE / 'B', C=CE / 'C', B_adapt=ROOT / 'B_adapt', C_adapt=ROOT / 'C_adapt')
for arm, path in paths.items():
    if not (path / 'results.json').exists():
        raise SystemExit('Incomplete: ' + arm + '; do not summarize target-best as final')
results = {arm: json.loads((path / 'results.json').read_text()) for arm, path in paths.items()}
comparisons = {}
for name, first, second in [('B_adapt-B', 'B_adapt', 'B'), ('C_adapt-C', 'C_adapt', 'C'), ('C-B', 'C', 'B'), ('C_adapt-B_adapt', 'C_adapt', 'B_adapt')]:
    comparisons[name] = {key: results[first]['fixed500'][key] - results[second]['fixed500'][key] for key in ['oa', 'aa', 'kappa']}
comparisons['interaction'] = {key: comparisons['C_adapt-C'][key] - comparisons['B_adapt-B'][key] for key in ['oa', 'aa', 'kappa']}
(ROOT / 'summary.json').write_text(json.dumps(dict(results=results, fixed500_deltas=comparisons, scope='single seed1233 diagnostic; no automatic model selection'), indent=2))
print(json.dumps(comparisons, indent=2))
