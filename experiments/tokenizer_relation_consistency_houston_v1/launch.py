import json,os,sys,time,subprocess
from common import OUT,SEEDS,dump

def main():
 assert json.loads((OUT/'preflight.json').read_text())['RI_preflight_passed']
 prior=OUT.parent/'tokenizer_mechanism_pavia_l2_fixed_v1/status.json'
 while True:
  s=json.loads(prior.read_text())
  if s['state']=='completed':break
  if any(x in s['state'] for x in ['failed','blocked']):
   dump(OUT/'status.json',{'state':'waiting_for_resource_review_after_Pavia_failure','prior':s,'no_auto_retry':True});return
  dump(OUT/'status.json',{'state':'queued_waiting_for_repaired_Pavia_GPU_resources','prior_state':s['state'],'Houston_C1_started':False,'C2':'blocked: no independent RE gradient','updated':time.time()});time.sleep(30)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 (OUT/'houston').mkdir(exist_ok=False);active=[]
 for gpu,seed in zip([0,4,5],SEEDS):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);log=(OUT/f'C1_{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','houston','--arm','C1','--seed',str(seed)],cwd=OUT,env=e,stdout=log,stderr=subprocess.STDOUT);active.append((gpu,seed,p,log))
 while active:
  done=[]
  for gpu,seed,p,log in active:
   if p.poll() is None:continue
   log.close()
   if p.returncode:
    dump(OUT/'status.json',{'state':'C1_training_failed','seed':seed,'exit_code':p.returncode,'no_auto_retry':True});return
   done.append(p.pid)
  active=[t for t in active if t[2].pid not in done]
  dump(OUT/'status.json',{'state':'C1_training','active':[{'gpu':g,'seed':s,'pid':p.pid} for g,s,p,l in active],'updated':time.time()})
  if active:time.sleep(15)
 with (OUT/'report.log').open('x') as log:code=subprocess.call([sys.executable,'-u',str(OUT/'report.py')],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 dump(OUT/'status.json',{'state':'completed' if code==0 else 'report_failed','exit_code':code,'C0':'reused official C 3seeds','C1_runs':3,'C2_runs':0,'Pavia_modified':False,'no_auto_followup':True,'updated':time.time()})
if __name__=='__main__':main()
