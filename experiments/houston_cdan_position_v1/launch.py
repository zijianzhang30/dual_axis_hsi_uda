import subprocess,sys,json,time,os
import study as s

def main():
 jobs=[]
 for arm,gpu in [('Z',6),('T',7)]:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');f=(s.OUT/('CDAN_'+arm+'.log')).open('x');p=subprocess.Popen([sys.executable,'-u',str(s.OUT/'code_snapshot/train.py'),'--arm',arm],env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((arm,gpu,p));print('START',arm,gpu,p.pid,flush=True)
 while any(p.poll() is None for a,g,p in jobs):
  s.dump(s.OUT/'status.json',{'status':'training_Z_T_seed2100','jobs':[{'arm':a,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for a,g,p in jobs]})
  if any(p.poll() is not None and p.returncode!=0 for a,g,p in jobs):
   s.dump(s.OUT/'status.json',{'status':'STOP_training_failure_no_automatic_retry'});return
  time.sleep(10)
 if any(p.returncode for a,g,p in jobs):return
 s.dump(s.OUT/'status.json',{'status':'offline_prediction_and_scoring'});env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (s.OUT/'report.log').open('x') as f:p=subprocess.run([sys.executable,'-u',str(s.OUT/'code_snapshot/report.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 s.dump(s.OUT/'status.json',{'status':'completed_no_followups' if p.returncode==0 else 'STOP_offline_report_failure','new_trainings':2,'Pavia_HyRANK_not_started':True})
if __name__=='__main__':main()
