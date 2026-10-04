"""Launch exactly two single-seed protocol diagnostics on GPUs 4 and 5."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1')
PROJECT = Path('/home/zhangzj26/dual_axis_hsi_uda')
ENV = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/env/bin/python')
assert not (ROOT / 'launch.json').exists(), 'Refusing duplicate tasks'
for arm in ['B', 'C']:
    assert (ROOT / arm / 'preflight.json').exists()
snap = ROOT / 'source_snapshot'
snap.mkdir(exist_ok=False)
for name in ['run.py', 'launch.py', 'PROTOCOL.md']:
    shutil.copy2(Path(__file__).with_name(name), snap / name)
shutil.copytree('/home/zhangzj26/IEEE_TCSVT_BiDA/models', snap / 'models', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(PROJECT / 'experiments/bn_stabilization_v1/normalization.py', snap / 'normalization.py')
shutil.copy2(PROJECT / 'experiments/relation_tokenizer_v1/relation.py', snap / 'relation.py')
manifest = {str(p.relative_to(snap)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()}
(ROOT / 'source_hashes.json').write_text(json.dumps(manifest, indent=2))
launch = []
for arm, gpu in [('B', 4), ('C', 5)]:
    command = [str(ENV), '-u', str(snap / 'run.py'), '--arm', arm]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (ROOT / arm / 'train.log').open('x') as f:
        proc = subprocess.Popen(command, env=environment, cwd=str(PROJECT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    launch.append(dict(arm=arm, gpu=gpu, pid=proc.pid, command=command))
(ROOT / 'launch.json').write_text(json.dumps(launch, indent=2))
print(json.dumps(launch))
