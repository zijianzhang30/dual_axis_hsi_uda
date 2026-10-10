import subprocess,sys,os,time
import common as c

def main():
 jobs=[]
 for seed,gpu in zip(c.SEEDS,(5,3,0)):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');f=(c.OUT/('seed_%d.log'%seed)).open('x');p=subprocess.Popen([sys.executable,'-u',str(c.OUT/'experiment_code/run.py'),'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((seed,gpu,p));print('DENSE_START',seed,gpu,p.pid,flush=True)
 while any(p.poll() is None for s,g,p in jobs):
  c.dump(c.OUT/'status.json',{'status':'full_scene_frozen_inference','jobs':[{'seed':s,'gpu':g,'pid':p.pid,'exit_code':p.poll()} for s,g,p in jobs]})
  if any(p.poll() is not None and p.returncode!=0 for s,g,p in jobs):c.dump(c.OUT/'status.json',{'status':'STOP_inference_failure_no_retry'});return
  time.sleep(10)
 if any(p.returncode for s,g,p in jobs):return
 c.dump(c.OUT/'status.json',{'status':'offline_GT_scoring'});env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (c.OUT/'report.log').open('x') as f:p=subprocess.run([sys.executable,'-u',str(c.OUT/'experiment_code/report.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 c.dump(c.OUT/'status.json',{'status':'completed_no_followups' if p.returncode==0 else 'STOP_offline_scoring_failure','no_training':True,'no_Pavia_HyRANK_SND':True})
if __name__=='__main__':main()
