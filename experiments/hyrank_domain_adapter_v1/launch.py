import os,json,subprocess,sys,time
from pathlib import Path
import study as s

def main():
 assert json.loads((s.OUT/'preflight.json').read_text())['passed'];s.verify()
 jobs=[]
 for arm,gpu in [('R1',5),('R2',5)]:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  log=(s.OUT/('%s_2100.log'%arm)).open('w')
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'run.py'),'--seed','2100','--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT)
  jobs.append((arm,gpu,p,log));print('START',arm,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'arm':a,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for a,gpu,p,_ in jobs]
  s.dump(s.OUT/'launcher_status.json',{'status':'running_fixed100_adapter_stage','jobs':rows,'R0_reused':True})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,log in jobs:log.close()
 if not all(r['exit_code']==0 for r in rows):
  s.dump(s.OUT/'launcher_status.json',{'status':'failed_no_automatic_retry','jobs':rows});return
 while not (s.R0/'C_2100/predictions/full_epoch_100.npz').exists():
  s.dump(s.OUT/'launcher_status.json',{'status':'adapters_complete_waiting_R0_epoch100_predictions','jobs':rows});time.sleep(20)
 with (s.OUT/'report.log').open('w') as f:subprocess.run([sys.executable,str(Path(__file__).parent/'report.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.OUT/'launcher_status.json',{'status':'completed_exploratory_no_automatic3seed','jobs':rows})
if __name__=='__main__':main()
