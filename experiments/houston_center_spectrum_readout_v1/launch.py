import os,sys,subprocess,time,json
from pathlib import Path
import study as s
CODE=Path(__file__).parent
JOBS=[('R1',2100,6),('R1',2101,7),('R1',2102,3),('R2',2100,4),('R2',2101,0),('R2',2102,1)]
def env(gpu):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');return e
def main():
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];jobs=[];evaluation_started=set()
 for arm,seed,gpu in JOBS:
  log=(s.OUT/('%s_%d.log'%(arm,seed))).open('x');p=subprocess.Popen([sys.executable,'-u',str(CODE/'train.py'),'--arm',arm,'--seed',str(seed)],env=env(gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append({'arm':arm,'seed':seed,'gpu':gpu,'process':p,'log':log,'stage':'train'});print('START_READOUT',arm,seed,gpu,p.pid,flush=True)
 while True:
  failure=False;active=False
  for j in jobs:
   p=j['process'];code=p.poll()
   if code is None:active=True;continue
   if code!=0:failure=True;continue
   key=(j['arm'],j['seed'])
   if j['stage']=='train' and key not in evaluation_started:
    j['process']=subprocess.Popen([sys.executable,'-u',str(CODE/'evaluate.py'),'--arm',j['arm'],'--seed',str(j['seed'])],env=env(j['gpu']),stdout=j['log'],stderr=subprocess.STDOUT);j['stage']='freeze_predictions';evaluation_started.add(key);active=True
  status=[{'arm':j['arm'],'seed':j['seed'],'gpu':j['gpu'],'pid':j['process'].pid,'stage':j['stage'],'exit_code':j['process'].poll()} for j in jobs];s.dump(s.OUT/'status.json',{'status':'running_six_readout_jobs' if active else ('failed_no_auto_retry' if failure else 'all6_predictions_frozen'), 'jobs':status,'new_training_limit':6,'target_scoring_deferred':True})
  if not active:break
  time.sleep(15)
 for j in jobs:j['log'].close()
 if any(j['process'].returncode!=0 for j in jobs):sys.exit(1)
 with (s.OUT/'report.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(CODE/'report.py')],env=env(6),stdout=f,stderr=subprocess.STDOUT)
 sys.exit(r.returncode)
if __name__=='__main__':main()
