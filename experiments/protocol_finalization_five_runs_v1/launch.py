import os,json,sys,time,subprocess
from pathlib import Path
import study as s

def main():
 s.verify();assert json.loads((s.TOP/'preflight.json').read_text())['passed'];jobs=[];code=Path(__file__).parent
 tasks=[('Houston_lr001',2100,5,'houston_train.py',['--dataset','houston','--arm','C','--seed','2100']),('CE',2101,0,'hyrank_train.py',['--arm','C','--seed','2101']),('LMMD',2101,1,'hyrank_train.py',['--arm','C','--seed','2101']),('CE',2102,2,'hyrank_train.py',['--arm','C','--seed','2102']),('LMMD',2102,7,'hyrank_train.py',['--arm','C','--seed','2102'])]
 for group,seed,gpu,script,args in tasks:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PROTOCOL_OBJECTIVE=group if group in ('CE','LMMD') else 'LMMD',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  f=(s.TOP/('%s_%d.log'%(group,seed))).open('x');p=subprocess.Popen([sys.executable,'-u',str(code/script)]+args,env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((group,seed,gpu,p,f));print('START',group,seed,'GPU',gpu,'PID',p.pid,flush=True)
 while True:
  rows=[{'group':g,'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for g,seed,gpu,p,_ in jobs];s.dump(s.TOP/'launcher_status.json',{'status':'running_five_predeclared_protocol_runs','jobs':rows})
  if all(r['exit_code'] is not None for r in rows):break
  time.sleep(10)
 for *_,f in jobs:f.close()
 if not all(r['exit_code']==0 for r in rows):s.dump(s.TOP/'launcher_status.json',{'status':'failed_no_automatic_retry','jobs':rows});return
 s.dump(s.TOP/'launcher_status.json',{'status':'offline_reporting_after_all_training','new_trainings':5})
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='5',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (s.TOP/'report.log').open('w') as f:subprocess.run([sys.executable,'-u',str(code/'report.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 s.verify();s.dump(s.TOP/'launcher_status.json',{'status':'completed_five_runs_no_followups','new_trainings':5,'HyRANK2100_reused':True,'Pavia_not_started':True})
if __name__=='__main__':main()
