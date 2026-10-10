import json,os,subprocess,sys,time
from pathlib import Path
import study as s
CODE=Path(__file__).parent
def env(gpu):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');return e
def main():
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];jobs=[]
 for arm,gpu in [('S1',3),('S2',6)]:
  f=(s.OUT/(arm+'_2100.log')).open('x');p=subprocess.Popen([sys.executable,'-u',str(CODE/'train.py'),'--arm',arm],env=env(gpu),stdout=f,stderr=subprocess.STDOUT);jobs.append({'arm':arm,'gpu':gpu,'p':p,'log':f,'stage':'train'});print('START_SCENE_ATTENTION',arm,gpu,p.pid,flush=True)
 while True:
  active=False
  for j in jobs:
   rc=j['p'].poll()
   if rc is None:active=True;continue
   if rc==0 and j['stage']=='train':j['p']=subprocess.Popen([sys.executable,'-u',str(CODE/'evaluate.py'),'--arm',j['arm']],env=env(j['gpu']),stdout=j['log'],stderr=subprocess.STDOUT);j['stage']='freeze_predictions';active=True
  rows=[{'arm':j['arm'],'gpu':j['gpu'],'pid':j['p'].pid,'stage':j['stage'],'exit_code':j['p'].poll()} for j in jobs];s.dump(s.OUT/'status.json',{'status':'running' if active else 'all_jobs_exited','jobs':rows,'target_scoring_deferred':True})
  if not active:break
  time.sleep(15)
 for j in jobs:j['log'].close()
 if any(j['p'].returncode!=0 for j in jobs):s.dump(s.OUT/'status.json',{'status':'stopped_failure_no_auto_retry_or_tuning','jobs':rows});sys.exit(1)
 with (s.OUT/'report.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(CODE/'report.py')],env=env(3),stdout=f,stderr=subprocess.STDOUT)
 sys.exit(r.returncode)
if __name__=='__main__':main()
