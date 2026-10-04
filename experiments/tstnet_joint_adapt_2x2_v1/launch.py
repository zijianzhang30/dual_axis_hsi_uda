import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_joint_adapt_2x2_v1')
assert not (ROOT / 'launch.json').exists()
for arm in ['B_adapt', 'C_adapt']:
    assert (ROOT / arm / 'preflight.json').exists()
records = []
for arm, gpu in [('B_adapt', 1), ('C_adapt', 2)]:
    command = ['/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/env/bin/python', '-u', str(ROOT / 'run.py'), '--arm', arm]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (ROOT / arm / 'train.log').open('x') as f:
        proc = subprocess.Popen(command, env=environment, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    records.append(dict(arm=arm, gpu=gpu, pid=proc.pid, command=command))
(ROOT / 'launch.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records))
