import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_anchor_fixed_v1')
ROOT.mkdir(parents=True, exist_ok=True)
assert not (ROOT / 'launch.json').exists()
(ROOT / 'bida_snapshot').symlink_to('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1/bida_snapshot', target_is_directory=True)
shutil.copytree('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_joint_adapt_2x2_v1/frozen_components', ROOT / 'frozen_components')
for name in ['run.py', 'anchors.py', 'verify.py', 'prepare.py', 'launch.py', 'PROTOCOL.md']:
    shutil.copy2(Path(__file__).with_name(name), ROOT / name)
(ROOT / 'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and 'bida_snapshot' not in p.parts}, indent=2))
print(ROOT)
