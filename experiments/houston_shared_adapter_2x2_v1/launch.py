import json,os,subprocess,sys,time
from pathlib import Path
import study as s

def env(gpu):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');return e

def main():
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];jobs=[];code=Path(__file__).parent
 for arm,seed,gpu in [('CE1',2100,5),('LMMD1',2100,4),('CE1',2101,1),('LMMD1',2101,7),('CE1',2102,0),('LMMD1',2102,3)]:
  f=(s.OUT/('%s_%d.log'%(arm,seed))).open('x');p=subprocess.Popen([sys.executable,'-u',str(code/'train.py'),'--arm',arm,'--seed',str(seed)],env=env(gpu),stdout=f,stderr=subprocess.STDOUT);jobs.append((arm,seed,gpu,p,f));print('START',arm,seed,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'group':arm,'seed':seed,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for arm,seed,g,p,_ in jobs];s.dump(s.OUT/'launcher_status.json',{'status':'training_six_Houston_shared_adapter_jobs','jobs':rows,'HyRANK_reused':True,'Pavia_deferred':True})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,f in jobs:f.close()
 if not all(r['exit_code']==0 for r in rows):s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retry','jobs':rows});return
 queues=[]
 for seed,g in [(2100,5),(2101,1),(2102,0)]:
  f=(s.OUT/('offline_%d.log'%seed)).open('w');commands=[[sys.executable,str(code/'evaluate.py'),'--group',arm,'--seed',str(seed)] for arm in s.GROUPS];script='import subprocess; commands='+repr(commands)+'; [subprocess.run(c,check=True) for c in commands]';p=subprocess.Popen([sys.executable,'-u','-c',script],env=env(g),stdout=f,stderr=subprocess.STDOUT);queues.append((seed,p,f))
 while True:
  rs=[{'seed':seed,'pid':p.pid,'exit_code':p.poll()} for seed,p,_ in queues];s.dump(s.OUT/'launcher_status.json',{'status':'offline_prediction_freezing','queues':rs})
  if all(r['exit_code'] is not None for r in rs):break
  time.sleep(10)
 for _,_,f in queues:f.close()
 assert all(r['exit_code']==0 for r in rs)
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,str(code/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.OUT/'launcher_status.json',{'status':'completed_Houston_2x2_HyRANK_reused_Pavia_deferred','new_trainings':6,'no_auto_followups':True})
if __name__=='__main__':main()
