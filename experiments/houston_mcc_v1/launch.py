import os,sys,json,time,subprocess
from pathlib import Path
import study as s

def env(gpu):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');return e

def main():
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];jobs=[];code=Path(__file__).parent
 for seed,gpu in [(2100,4),(2101,1),(2102,3)]:
  f=(s.OUT/('seed_%d.log'%seed)).open('x');p=subprocess.Popen([sys.executable,'-u',str(code/'train.py'),'--seed',str(seed)],env=env(gpu),stdout=f,stderr=subprocess.STDOUT);jobs.append((seed,gpu,p,f));print('START_MCC',seed,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'seed':seed,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for seed,g,p,_ in jobs];s.dump(s.OUT/'status.json',{'status':'training_three_MCC_runs','jobs':rows})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,f in jobs:f.close()
 if not all(r['exit_code']==0 for r in rows):s.dump(s.OUT/'status.json',{'status':'STOP_failure_no_automatic_loss_change_or_retry','jobs':rows});return
 s.dump(s.OUT/'status.json',{'status':'offline_prediction_and_scoring','new_trainings':3})
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,'-u',str(code/'report.py')],env=env(4),stdout=f,stderr=subprocess.STDOUT,check=True)
 s.dump(s.OUT/'status.json',{'status':'completed_MCC_exploratory_no_followups','new_trainings':3,'Pavia_HyRANK_not_started':True})
if __name__=='__main__':main()
