import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_anchor_fixed_v1')
assert (ROOT / 'geometry_verification.json').exists()
assert (ROOT / 'strict_gpu_verification.json').exists()
assert not (ROOT / 'launch.json').exists()
for arm in ['C', 'C_adapt']:
    assert (ROOT / arm / 'preflight.json').exists()
records = []
for arm, gpu in [('C', 5), ('C_adapt', 7)]:
    command = ['/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/env/bin/python', '-u', str(ROOT / 'run.py'), '--arm', arm]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (ROOT / arm / 'train.log').open('x') as f:
        process = subprocess.Popen(command, env=environment, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    records.append(dict(arm=arm, gpu=gpu, pid=process.pid, command=command))
(ROOT / 'launch.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records))
