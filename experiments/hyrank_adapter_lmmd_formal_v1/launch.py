import json,os,subprocess,sys,time
from pathlib import Path
import study as s

def env(gpu):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');return e

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify();jobs=[]
 for arm,seed,gpu in [('R1',2100,1),('R2',2100,3),('R1',2101,6),('R2',2101,7),('R1',2102,2),('R2',2102,4)]:
  log=(s.OUT/('%s_%d.log'%(arm,seed))).open('x');p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'train.py'),'--arm',arm,'--seed',str(seed)],env=env(gpu),stdout=log,stderr=subprocess.STDOUT)
  jobs.append((arm,seed,gpu,p,log));print('START',arm,seed,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'arm':a,'seed':seed,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for a,seed,g,p,_ in jobs]
  s.dump(s.OUT/'launcher_status.json',{'status':'training_six_adapters_historical_LMMD_recipe','jobs':rows,'R0_reused':True})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,log in jobs:log.close()
 if not all(r['exit_code']==0 for r in rows):s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retries','jobs':rows});return
 # Three sequential inference queues in parallel; all outputs frozen before GT scoring.
 queues=[]
 for seed,gpu in [(2100,1),(2101,6),(2102,2)]:
  log=(s.OUT/('offline_eval_%d.log'%seed)).open('w')
  commands=[[sys.executable,str(Path(__file__).parent/'evaluate.py'),'--seed',str(seed),'--arm',a] for a in ('R0','R1','R2')]
  # Run each seed queue in a child Python process with explicit argv, no shell.
  script='import subprocess; commands='+repr(commands)+'; [subprocess.run(c,check=True) for c in commands]'
  p=subprocess.Popen([sys.executable,'-u','-c',script],env=env(gpu),stdout=log,stderr=subprocess.STDOUT);queues.append((seed,p,log))
 while True:
  q=[{'seed':seed,'pid':p.pid,'exit_code':p.poll()} for seed,p,_ in queues];s.dump(s.OUT/'launcher_status.json',{'status':'offline_inference_only','queues':q})
  if all(r['exit_code'] is not None for r in q):break
  time.sleep(10)
 for _,_,log in queues:log.close()
 assert all(r['exit_code']==0 for r in q)
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).parent/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.OUT/'launcher_status.json',{'status':'completed_formal_adapter_study_no_followups','new_trainings':6,'R0_reused':True})
if __name__=='__main__':main()
