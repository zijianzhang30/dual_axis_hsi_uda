"""Start the single native 10-trial sequence without stopping any other jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_hyrank_native_v1')
assert (ROOT / 'config.json').exists() and (ROOT / 'preflight.json').exists()
assert not (ROOT / 'launch.json').exists(), 'Refusing duplicate launch'
script = Path(__file__).with_name('run.py')
shutil.copy2(script, ROOT / 'run.py')
command = [str(ROOT / 'env/bin/python'), '-u', str(ROOT / 'run.py')]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
with (ROOT / 'train.log').open('x') as log:
    process = subprocess.Popen(command, cwd=str(ROOT / 'run_snapshot'), env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=process.pid, gpu=0, command=command, runner_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
(ROOT / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
