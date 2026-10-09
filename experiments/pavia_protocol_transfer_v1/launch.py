"""Exactly six runs on idle GPUs0/7, then fixed-endpoint scoring; no retries."""
import json
import os
import subprocess
import sys
import time
import common as c


def main():
    assert json.loads((c.OUT/'verification.json').read_text())['passed']
    assert not (c.OUT/'launcher_status.json').exists(),'Refuse duplicate launch'
    jobs=[]
    for seed in c.SEEDS:
        for arm,gpu in (('P1',0),('P2',7)):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            log=(c.OUT/('%s_%d.log'%(arm,seed))).open('x')
            p=subprocess.Popen([sys.executable,'-u',str(c.OUT/'train.py'),'--arm',arm,'--seed',str(seed)],
                               cwd=c.OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append({'process':p,'log':log,'arm':arm,'seed':seed,'gpu':gpu})
    def status(state):
        c.dump(c.OUT/'launcher_status.json',{'state':state,'updated':time.time(),'jobs':[
            {'arm':r['arm'],'seed':r['seed'],'gpu':r['gpu'],'pid':r['process'].pid,'exit_code':r['process'].poll()} for r in jobs]})
    status('training')
    while any(r['process'].poll() is None for r in jobs):
        time.sleep(15);status('training')
    for r in jobs:r['log'].close()
    if any(r['process'].returncode!=0 for r in jobs):
        status('failed_no_restart_no_scoring');raise RuntimeError('Run failed; no automatic repair/restart')
    status('all_predictions_frozen_scoring')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run([sys.executable,str(c.OUT/'summarize.py')],cwd=c.OUT,env=env,check=True)
    status('completed_no_followups')


if __name__=='__main__':main()
