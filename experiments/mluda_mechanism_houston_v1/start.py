"""Finalize study sources then detach the fixed queue, preserving existing jobs."""
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

from common import OUT, dump, sha

assert json.loads((OUT/'verification.json').read_text())['passed']
runpy.run_path(str(Path(__file__).with_name('refresh.py')),run_name='__main__')
old=json.loads((OUT/'baseline_environment.json').read_text())
versions={name:importlib.metadata.version(name) for name,value in old['packages'].items() if value is not None}
assert versions=={name:value for name,value in old['packages'].items() if value is not None}
env=os.environ.copy()
env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
with (OUT/'launcher.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=str(OUT),env=env,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'started_at':__import__('time').time(),
    'physical_gpus':[0,1,6,7],'tasks':12,'environment_packages':versions,
    'verification_sha256':sha(OUT/'verification.json')})
print('Started fixed12-task queue PID',p.pid,flush=True)
