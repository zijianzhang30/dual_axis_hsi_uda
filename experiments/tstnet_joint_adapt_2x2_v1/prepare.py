import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_joint_adapt_2x2_v1')
PROJECT = Path('/home/zhangzj26/dual_axis_hsi_uda')
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'bida_snapshot').symlink_to('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1/bida_snapshot', target_is_directory=True)
components = ROOT / 'frozen_components'
components.mkdir(exist_ok=False)
for name, path in [('normalization.py', PROJECT / 'experiments/bn_stabilization_v1/normalization.py'), ('relation.py', PROJECT / 'experiments/relation_tokenizer_v1/relation.py')]:
    shutil.copy2(path, components / name)
for name in ['run.py', 'prepare.py', 'launch.py', 'summarize.py', 'PROTOCOL.md']:
    shutil.copy2(Path(__file__).with_name(name), ROOT / name)
for arm in ['B', 'C']:
    shutil.copy2(Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1') / arm / 'config.json', ROOT / (arm + '_CE_frozen_config.json'))
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and 'bida_snapshot' not in p.parts}
(ROOT / 'source_hashes.json').write_text(json.dumps(manifest, indent=2))
print(ROOT)
