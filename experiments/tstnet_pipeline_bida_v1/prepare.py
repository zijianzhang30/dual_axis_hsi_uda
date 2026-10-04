"""Archive dependencies for the new task, without moving active dependencies."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1')
ROOT.mkdir(parents=True, exist_ok=True)
target = ROOT / 'bida_snapshot'
target.mkdir(exist_ok=False)
original = Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
for name in ['models', 'utils', 'loss']:
    shutil.copytree(original / name, target / name, ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(original / 'train_pipeline.py', target / 'train_pipeline.py')
manifest = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
(ROOT / 'bida_hashes.json').write_text(json.dumps(manifest, indent=2))
for name in ['run.py', 'prepare.py', 'launch.py', 'PROTOCOL.md']:
    shutil.copy2(Path(__file__).with_name(name), ROOT / name)
print(str(ROOT))
