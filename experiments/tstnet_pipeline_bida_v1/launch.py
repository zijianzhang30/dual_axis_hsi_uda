import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1')
assert (ROOT / 'A/preflight.json').exists()
assert not (ROOT / 'launch.json').exists()
command = ['/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1/env/bin/python', '-u', str(ROOT / 'run.py'), '--arm', 'A']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
with (ROOT / 'A/train.log').open('x') as f:
    proc = subprocess.Popen(command, env=env, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=proc.pid, gpu=3, command=command)
(ROOT / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
