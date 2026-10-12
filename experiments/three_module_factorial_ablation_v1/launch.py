import json,os,subprocess,sys,time
from pathlib import Path
import study as s
def resources():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();return [(int(a),int(b),int(c)) for a,b,c in (line.split(',') for line in lines)]
class AdoptedProcess:
 def __init__(self,record):self.record=record
 def poll(self):
  record=self.record;status=Path('/proc/%d/status'%record['pid'])
  if status.exists():
   try:
    if '\nState:\tZ' not in status.read_text():return None
   except FileNotFoundError:pass
  marker=s.OUT/record['dataset']/('%s_%d'%(record['relation'],record['seed']))/'TRAINING_PREDICTIONS_FROZEN.json'
  return 0 if marker.exists() else -1
def main():
 s.verify();assert all(json.loads((s.OUT/('preflight_'+d+'.json')).read_text())['passed'] for d in s.DATASETS)
 if '--resume' in sys.argv:
  prior=json.loads((s.OUT/'STATUS.json').read_text());pending=prior['pending'];done=prior['completed'];failed=prior['failed_no_auto_retry'];active=[]
  for record in prior['active']:
   item=dict(record);item['process']=AdoptedProcess(item);item['log']=None;active.append(item)
 else:
  pending=s.new_tasks();assert len(pending)==21;active=[];done=[];failed=[]
 assert len(pending)+len(active)+len(done)+len(failed)==21
 start=time.time();s.dump(s.OUT/'launcher_pid.json',{'pid':os.getpid(),'started_unix':start,'resumed':'--resume' in sys.argv})
 # HyRANK first: supplies the missing clean same-protocol ablations discussed.
 pending.sort(key=lambda a:(0 if a[0]=='hyrank' else 1 if a[0]=='houston' else 2,a[1],a[2]))
 while pending or active:
  for item in active[:]:
   rc=item['process'].poll()
   if rc is None:continue
   active.remove(item);item['log'].close() if item['log'] is not None else None;record={k:v for k,v in item.items() if k not in ('process','log')};record['exit_code']=rc;(done if rc==0 else failed).append(record)
  while pending and len(active)<8:
   eligible=[(util,gpu) for gpu,free,util in resources() if gpu in range(8) and free>=20480 and sum(a['gpu']==gpu for a in active)<1]
   if not eligible:break
   gpu=min(eligible)[1]
   d,r,seed=pending.pop(0);logpath=s.OUT/('train_%s_%s_%d.log'%(d,r,seed));log=logpath.open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2');cmd=[sys.executable,str(s.CODE/'train.py'),'--dataset',d,'--relation',r,'--seed',str(seed)];p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=str(s.ROOT));active.append({'dataset':d,'relation':r,'seed':seed,'gpu':gpu,'pid':p.pid,'started_unix':time.time(),'log_path':str(logpath),'process':p,'log':log});print('LAUNCHED',d,r,seed,gpu,p.pid,flush=True)
  s.dump(s.OUT/'STATUS.json',{'state':'running' if pending or active else 'training_done' if not failed else 'completed_with_failures','planned_new_trainings':21,'pending':pending,'active':[{k:v for k,v in a.items() if k not in ('process','log')} for a in active],'completed':done,'failed_no_auto_retry':failed,'updated_unix':time.time(),'no_unplanned_jobs':True});time.sleep(10 if pending or active else 0)
 if failed:print('STOP:failures recorded; no retries or target scoring',flush=True);return
 s.verify();print('ALL21_TRAININGS_AND_PREDICTIONS_FROZEN',flush=True)
 # Only the already-planned, label-isolated scoring/reporting follows. No new training.
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
 with (s.OUT/'report.log').open('x') as f:rc=subprocess.call([sys.executable,str(s.CODE/'report.py')],cwd=str(s.ROOT),env=env,stdout=f,stderr=subprocess.STDOUT)
 s.dump(s.OUT/'COMPLETION_STATUS.json',{'all21_trainings_passed':True,'report_exit_code':rc,'no_followup_started':True});print('REPORT_FINISHED',rc,flush=True)
if __name__=='__main__':main()
