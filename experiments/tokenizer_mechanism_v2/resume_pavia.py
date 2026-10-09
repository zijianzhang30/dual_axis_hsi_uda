"""Exactly Houston18 runs; Pavia18 only if declared result gate passes; no retries."""
import json
import os
import subprocess
import sys
import time
from common import OUT,ARMS,SEEDS,dump
GPUS=(0,1,4,5)

def stage(dataset):
 (OUT/dataset).mkdir(exist_ok=False);pending=[(a,s) for s in SEEDS for a in ARMS];active={};completed=[]
 while pending or active:
  for gpu,(p,a,s,log) in list(active.items()):
   code=p.poll()
   if code is None:continue
   log.close();del active[gpu]
   if code!=0:
    dump(OUT/'status.json',{'state':'failed','dataset':dataset,'arm':a,'seed':s,'exit_code':code,'active':[{'pid':q.pid,'arm':b,'seed':v,'gpu':g} for g,(q,b,v,l) in active.items()],'pending':pending,'completed':completed})
    raise RuntimeError('Failure, no automatic retry/other training')
   completed.append([a,s])
  for gpu in GPUS:
   if gpu in active or not pending:continue
   a,s=pending.pop(0);env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
   log=(OUT/dataset/f'{a}_{s}.log').open('x')
   p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset',dataset,'--arm',a,'--seed',str(s)],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
   active[gpu]=(p,a,s,log)
  dump(OUT/'status.json',{'state':'training','dataset':dataset,'pending':pending,'completed':completed,'active':[{'gpu':g,'pid':p.pid,'arm':a,'seed':s} for g,(p,a,s,l) in active.items()],'updated':time.time()})
  if active:time.sleep(15)
 with (OUT/dataset/'summary.log').open('x') as log:
  code=subprocess.call([sys.executable,'-u',str(OUT/'summarize_review.py'),'--dataset',dataset],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
 if code!=0:
  dump(OUT/'status.json',{'state':'audit_or_scoring_failed','dataset':dataset});raise RuntimeError('Audit/scoring failed')
 return json.loads((OUT/dataset/'continuation_decision.json').read_text())

def analyze_weights(dataset):
 env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (OUT/dataset/'weight_analysis.log').open('x') as log:
  p=subprocess.Popen([sys.executable,'-u',str(OUT/'weight_analysis.py'),'--dataset',dataset],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 return p

def main():
 decision=json.loads((OUT/'houston/continuation_decision.json').read_text())
 assert decision['continue_pavia']
 assert (OUT/'houston/all_predictions_frozen.json').exists()
 precision=json.loads((OUT/'houston/linear_precision_review.json').read_text())
 assert max(r['same_parameters_FP64_max_weight_error'] for r in precision['runs'].values())<1e-10
 stage('pavia')
 with (OUT/'pavia/precision_review.log').open('x') as log:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
  code=subprocess.call([sys.executable,'-u',str(OUT/'linear_precision_review.py'),'--dataset','pavia'],cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert code==0
 p=analyze_weights('pavia')
 dump(OUT/'status.json',{'state':'analyzing_final_weights','Pavia_started':True,'updated':time.time()})
 assert p.wait()==0
 # Houston diagnostic may still be finishing in the independent read-only process.
 while not (OUT/'houston/weight_analysis.json').exists():time.sleep(15)
 with (OUT/'final_report.log').open('x') as log:
  code=subprocess.call([sys.executable,'-u',str(OUT/'final_report.py')],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
 dump(OUT/'status.json',{'state':'completed' if code==0 else 'report_failed','Houston_runs':18,'Pavia_runs':18,'HyRANK_training_runs':0,'updated':time.time(),'precision_review':'FP64 same-parameter equivalence; FP32 errors disclosed'})
if __name__=='__main__':main()
