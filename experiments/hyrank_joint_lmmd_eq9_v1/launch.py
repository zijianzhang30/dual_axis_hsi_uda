import json,os,subprocess,sys,time
from pathlib import Path
import study as s

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify();jobs=[]
 for arm,seed,gpu in [('A',2100,0),('C',2100,3),('A',2101,6),('C',2101,7),('A',2102,2),('C',2102,4)]:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  log=(s.OUT/('%s_%d.log'%(arm,seed))).open('w')
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'run.py'),'--arm',arm,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((arm,seed,gpu,p,log));print('START',arm,seed,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'arm':a,'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for a,seed,gpu,p,_ in jobs]
  s.dump(s.OUT/'launcher_status.json',{'status':'running_unified_A_C_recipe','jobs':rows,'training_count':6,'target_scoring_during_training':False})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,log in jobs:log.close()
 if not all(r['exit_code']==0 for r in rows):
  s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retries','jobs':rows});raise RuntimeError('No extra training; inspect failure')
 with (s.OUT/'offline_report.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).parent/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.OUT/'launcher_status.json',{'status':'completed_offline_report_no_followups','jobs':rows,'training_count':6})
if __name__=='__main__':main()
