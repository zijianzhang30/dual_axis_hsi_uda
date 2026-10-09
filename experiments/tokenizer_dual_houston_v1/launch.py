import os,sys,json,time,subprocess
from common import OUT,SEEDS,dump

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed']
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 active=[];(OUT/'houston').mkdir(exist_ok=False)
 for gpu,seed in zip([3,6,7],SEEDS):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);log=(OUT/f'H_{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','houston','--arm','H','--seed',str(seed)],cwd=OUT,env=e,stdout=log,stderr=subprocess.STDOUT);active.append((gpu,seed,p,log))
 while active:
  done=[]
  for gpu,s,p,l in active:
   if p.poll() is None:continue
   l.close()
   if p.returncode:dump(OUT/'status.json',{'state':'training_failed','seed':s,'exit_code':p.returncode,'no_auto_retry':True});return
   done.append(p.pid)
  active=[q for q in active if q[2].pid not in done]
  dump(OUT/'status.json',{'state':'training','active':[{'gpu':g,'seed':s,'pid':p.pid} for g,s,p,l in active],'updated':time.time()})
  if active:time.sleep(15)
 dump(OUT/'status.json',{'state':'evaluating_all_H_epoch200','updated':time.time()})
 with (OUT/'report.log').open('x') as log:code=subprocess.call([sys.executable,'-u',str(OUT/'report.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 dump(OUT/'status.json',{'state':'completed' if code==0 else 'report_failed','exit_code':code,'H_runs':3,'epochs':200,'A_C_reused':True,'no_automatic_followup':True,'updated':time.time()})
if __name__=='__main__':main()
