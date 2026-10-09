"""Three independent read-only GPU workers; GT reporting only after all finish."""
import json
import os
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).parent
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_subprototype_diagnosis_v1')
jobs=[];state={}
for seed,gpu in zip((2100,2101,2102),(0,2,5)):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log=(OUT/('seed_%d.log'%seed)).open('w')
    p=subprocess.Popen([sys.executable,'-u',str(HERE/'diagnose.py'),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env=env)
    jobs.append((seed,p,log));state[str(seed)]={'pid':p.pid,'gpu':gpu}
(OUT/'launch_state.json').write_text(json.dumps({'status':'running_readonly','workers':state},indent=2)+'\n')
for seed,p,log in jobs:
    code=p.wait();log.close();state[str(seed)]['exit_code']=code
    print('worker %d exit %d'%(seed,code),flush=True)
assert all(v['exit_code']==0 for v in state.values()),state
with (OUT/'summary.log').open('w') as log:
    subprocess.run([sys.executable,'-u',str(HERE/'diagnose.py'),'--summarize'],stdout=log,stderr=subprocess.STDOUT,check=True)
(OUT/'launch_state.json').write_text(json.dumps({'status':'completed_no_followups','workers':state},indent=2)+'\n')
print('All read-only diagnostics completed; no follow-ups launched.',flush=True)
