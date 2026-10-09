"""C first, A second; stages predeclared and unconditional; score after all6 freeze."""
import json
import os
import subprocess
import sys
import time
import common as c

def main():
    assert json.loads((c.OUT/'verification.json').read_text())['passed']
    assert not (c.OUT/'launcher_status.json').exists(),'Refuse duplicate launch'
    c.verify_files();jobs=[]
    def status(state):
        c.dump(c.OUT/'launcher_status.json',{'state':state,'updated':time.time(),'jobs':[
            {'arm':j['arm'],'seed':j['seed'],'gpu':j['gpu'],'pid':j['proc'].pid,'exit_code':j['proc'].poll()} for j in jobs]})
    for arm in c.NEW_ARMS:
        stage=[]
        for seed,gpu in zip(c.SEEDS,(5,6,6)):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
            log=(c.OUT/('%s_%d.log'%(arm,seed))).open('x')
            proc=subprocess.Popen([sys.executable,'-u',str(c.OUT/'train.py'),'--arm',arm,'--seed',str(seed)],
                cwd=c.OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
            job={'proc':proc,'log':log,'arm':arm,'seed':seed,'gpu':gpu};jobs.append(job);stage.append(job)
        status('training_'+arm)
        while any(j['proc'].poll() is None for j in stage):time.sleep(10);status('training_'+arm)
        for j in stage:j['log'].close()
        if any(j['proc'].returncode!=0 for j in stage):
            status('failed_no_restart_no_scoring');raise RuntimeError('Failure;no auto retries or protocol changes')
        status('predictions_frozen_'+arm)
    status('all6_predictions_frozen_scoring')
    subprocess.run([sys.executable,str(c.OUT/'summarize.py')],cwd=c.OUT,check=True)
    status('completed_no_followups')

if __name__=='__main__':main()
