"""Freeze final new study sources after preflight, detach bounded study queue."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import OUT,dump,sha
assert not (OUT/'status.json').exists()
assert json.loads((OUT/'verification.json').read_text())['passed']
provenance=json.loads((OUT/'provenance.json').read_text())
for p in Path(__file__).parent.iterdir():
 if p.is_file():shutil.copy2(p,OUT/p.name)
provenance['study_sha256']={p.name:sha(p) for p in OUT.glob('*.py')};provenance['protocol_sha256']=sha(OUT/'PROTOCOL.md')
dump(OUT/'provenance.json',provenance)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='2')
with (OUT/'launcher.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-u',str(OUT/'launch.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
dump(OUT/'launch.json',{'pid':p.pid,'physical_gpus':[0,1,4,5],'started':time.time(),'initial_stage':'Houston18 runs','Pavia':'only if declared Houston result gate passes','HyRANK':'no training','no_automatic_retry':True})
print('Started controller',p.pid,flush=True)
