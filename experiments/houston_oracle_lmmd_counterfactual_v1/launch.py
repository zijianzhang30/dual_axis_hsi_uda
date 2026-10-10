import sys,os,subprocess,time
import common as s

def main():
 jobs=[]
 for seed,gpu in zip(s.SEEDS,(4,7,3)):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');f=(s.OUT/('seed_%d.log'%seed)).open('x');p=subprocess.Popen([sys.executable,'-u',str(s.OUT/'code_snapshot/run.py'),'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((seed,gpu,p));print('COUNTERFACTUAL_START',seed,gpu,p.pid,flush=True)
 while any(p.poll() is None for seed,gpu,p in jobs):
  s.dump(s.OUT/'status.json',{'status':'isolated_one_step_Oracle_diagnostic','jobs':[{'seed':seed,'gpu':gpu,'pid':p.pid,'exit_code':p.poll()} for seed,gpu,p in jobs],'new_training_runs':0})
  if any(p.poll() is not None and p.returncode for seed,gpu,p in jobs):s.dump(s.OUT/'status.json',{'status':'STOP_counterfactual_failure_no_retry'});return
  time.sleep(10)
 if any(p.returncode for seed,gpu,p in jobs):return
 s.dump(s.OUT/'status.json',{'status':'offline_heldout_scoring','new_training_runs':0});env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (s.OUT/'report.log').open('x') as f:p=subprocess.run([sys.executable,'-u',str(s.OUT/'code_snapshot/report.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:s.dump(s.OUT/'status.json',{'status':'STOP_offline_report_failure','returncode':p.returncode})
if __name__=='__main__':main()
