"""Three independent read-only seed workers; no retries or followup training."""
import os
import subprocess
import sys
import time
import diagnostics as d

def main():
    assert not (d.OUT/'launcher_status.json').exists(),'Refuse duplicate launch';d.verify_files();jobs=[]
    for seed,gpu in zip(d.SEEDS,(5,6,6)):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        log=(d.OUT/('seed_%d.log'%seed)).open('x')
        proc=subprocess.Popen([sys.executable,'-u',str(d.OUT/'run.py'),'--seed',str(seed)],cwd=d.OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append({'proc':proc,'log':log,'seed':seed,'gpu':gpu})
    def status(state):d.write(d.OUT/'launcher_status.json',{'state':state,'updated':time.time(),'jobs':[
        {'seed':j['seed'],'gpu':j['gpu'],'pid':j['proc'].pid,'exit_code':j['proc'].poll()} for j in jobs]})
    status('read_only_diagnosis')
    while any(j['proc'].poll() is None for j in jobs):time.sleep(10);status('read_only_diagnosis')
    for j in jobs:j['log'].close()
    if any(j['proc'].returncode!=0 for j in jobs):status('failed_no_training');raise RuntimeError('Read-only audit failed;no automatic training')
    status('summarizing');subprocess.run([sys.executable,str(d.OUT/'summarize.py')],cwd=d.OUT,check=True)
    status('completed_no_followups')

if __name__=='__main__':main()
