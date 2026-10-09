import os,sys,json,time,subprocess
from common import OUT,SEEDS,dump

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed'];(OUT/'houston').mkdir(exist_ok=False)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');active=[]
 for seed,gpu in zip(SEEDS,[3,6,7]):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);log=(OUT/f'CQ1_{seed}_e200.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','houston','--arm','CQ1','--seed',str(seed),'--epochs','200'],cwd=OUT,env=e,stdout=log,stderr=subprocess.STDOUT);active.append((seed,gpu,p,log))
 while active:
  finished=[]
  for seed,gpu,p,log in active:
   if p.poll() is None:continue
   log.close()
   if p.returncode:dump(OUT/'status.json',{'state':'training_failed','seed':seed,'exit_code':p.returncode,'no_auto_retry':True});return
   finished.append(p.pid)
  active=[r for r in active if r[2].pid not in finished];dump(OUT/'status.json',{'state':'training','active':[{'seed':s,'gpu':g,'pid':p.pid,'epoch_budget':200} for s,g,p,l in active],'updated':time.time()})
  if active:time.sleep(10)
 e=env.copy();e['CUDA_VISIBLE_DEVICES']='6';dump(OUT/'status.json',{'state':'evaluating_fixed200','updated':time.time()})
 with (OUT/'evaluate.log').open('x') as log:code=subprocess.call([sys.executable,'-u',str(OUT/'evaluate.py')],cwd=OUT,env=e,stdout=log,stderr=subprocess.STDOUT)
 dump(OUT/'status.json',{'state':'completed' if code==0 else 'evaluation_failed','exit_code':code,'seeds':list(SEEDS),'epochs':200,'Pavia_modified':False,'target_free_not_launched_by_this_controller':True,'updated':time.time()})
if __name__=='__main__':main()
