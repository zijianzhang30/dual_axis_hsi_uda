"""Wait for A-F untouched; summarize it first; then only authorized G stages."""
import json
import os
import subprocess
import sys
import time
from common import BASE,OUT,SEEDS,dump
GPUS=(0,4,5)

def call(script,args,logname):
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 with (OUT/logname).open('x') as log:
  code=subprocess.call([sys.executable,'-u',str(OUT/script),*args],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if code:raise RuntimeError('Independent script failed: '+script)

def stage(dataset):
 root=OUT/dataset;root.mkdir(exist_ok=False);pending=list(SEEDS);active={};completed=[]
 while pending or active:
  for gpu,(p,seed,log) in list(active.items()):
   code=p.poll()
   if code is None:continue
   log.close();del active[gpu]
   if code:
    dump(OUT/'status.json',{'state':'G_training_failed','dataset':dataset,'seed':seed,'exit_code':code,'remaining_active':[q.pid for q,s,l in active.values()],'no_automatic_retry':True});raise RuntimeError('G failed')
   completed.append(seed)
  for gpu in GPUS:
   if gpu in active or not pending:continue
   seed=pending.pop(0);env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
   log=(root/f'G_{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset',dataset,'--arm','G','--seed',str(seed)],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=(p,seed,log)
  dump(OUT/'status.json',{'state':'G_training','dataset':dataset,'pending':pending,'completed':completed,'active':[{'gpu':g,'pid':p.pid,'seed':s} for g,(p,s,l) in active.items()],'updated':time.time()})
  if active:time.sleep(15)
 call('summarize.py',['--dataset',dataset],dataset+'/summary.log')
 return json.loads((OUT/dataset/'summary.json').read_text())['continuation']

def main():
 assert json.loads((OUT/'verification.json').read_text())['passed']
 # No child/G training until the existing A-F controller has completed itself.
 while True:
  status=json.loads((BASE/'status.json').read_text())
  if status['state']=='completed':break
  if any(word in status['state'] for word in ['failed','error']):
   dump(OUT/'status.json',{'state':'stopped_for_A_F_failure_review','original_status':status,'G_started':False});return
  dump(OUT/'status.json',{'state':'waiting_for_Pavia_A_F_completion_and_summary','original_state':status['state'],'original_dataset':status.get('dataset'),'G_started':False,'updated':time.time()})
  time.sleep(30)
 assert (BASE/'pavia/summary.json').exists() and (BASE/'pavia/weight_analysis.json').exists()
 call('summarize.py',['--AF'],'A_F_summary.log')
 pavia=json.loads((BASE/'pavia/summary.json').read_text())
 supports=any(pavia['continuation']['gates'][name]['passed'] for name in ['C-B','C-E','C-F'])
 if not supports:
  call('final_report.py',[],'final_report.log')
  dump(OUT/'status.json',{'state':'stopped_after_negative_Pavia_A_F','G_started':False,'independent_A_F_summary':str(OUT/'A_F_SUMMARY.txt'),'no_new_module_designed':True});return
 decision=stage('houston')
 # Analyze Houston G read-only while the conditionally authorized Pavia stage runs.
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 log=(OUT/'houston/weight_analysis.log').open('x')
 diagnostic=subprocess.Popen([sys.executable,'-u',str(OUT/'weight_analysis.py'),'--dataset','houston'],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if decision['continue_Pavia_G']:
  stage('pavia');call('weight_analysis.py',['--dataset','pavia'],'pavia/weight_analysis.log')
 dump(OUT/'status.json',{'state':'analyzing_G_weights','Pavia_G_started':decision['continue_Pavia_G'],'updated':time.time()})
 assert diagnostic.wait()==0;log.close()
 call('final_report.py',[],'final_report.log')
 dump(OUT/'status.json',{'state':'completed','Houston_G_runs':3,'Pavia_G_runs':3 if decision['continue_Pavia_G'] else 0,'HyRANK_runs':0,'original_A_F_unchanged':True,'updated':time.time()})
if __name__=='__main__':main()
