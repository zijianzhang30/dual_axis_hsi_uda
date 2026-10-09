"""Fixed four-GPU queue; score each benchmark after all six final endpoints exist."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import common as c

HERE=Path(__file__).parent
def env(gpu):
    v=os.environ.copy();v.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');return v

def main():
    # All preflights are source/target-label-blind probes, before any official training.
    for ds in c.PLAN['datasets']:
      for seed in c.SEEDS:
        file=c.OUT/('preflight_%s_%d.json'%(ds,seed))
        if file.exists():assert json.loads(file.read_text())['passed'];continue
        with (c.OUT/('preflight_%s_%d.log'%(ds,seed))).open('w') as log:
            subprocess.run([sys.executable,'-u',str(HERE/'train.py'),'--dataset',ds,'--seed',str(seed),'--preflight'],stdout=log,stderr=subprocess.STDOUT,env=env(2),check=True)
    subprocess.run([sys.executable,str(HERE/'prepare.py')],check=True,env=env(2))
    tasks=[(ds,arm,seed) for ds in c.PLAN['datasets'] for seed in c.SEEDS for arm in ('A','C')]
    devices=[0,2,4,5];running={};finished=[];scored=[];state={'status':'running','completed':finished,'scored':scored,'queue':tasks,'running':{}}
    def save():
        state['running']={str(gpu):{'dataset':v[1][0],'arm':v[1][1],'seed':v[1][2],'pid':v[0].pid} for gpu,v in running.items()}
        c.dump(c.OUT/'launch_state.json',state)
    while tasks or running:
        for gpu in devices:
            if gpu in running or not tasks:continue
            ds,arm,seed=tasks.pop(0);log=(c.OUT/('%s_%s_%d.log'%(ds,arm,seed))).open('w')
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--dataset',ds,'--arm',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env=env(gpu))
            running[gpu]=(p,(ds,arm,seed),log);print('Started %s %s seed%d on GPU%d'%(ds,arm,seed,gpu),flush=True)
        for gpu,(p,task,log) in list(running.items()):
            code=p.poll()
            if code is None:continue
            log.close();finished.append({'dataset':task[0],'arm':task[1],'seed':task[2],'exit_code':code});del running[gpu]
            print('Finished %s exit%d'%(task,code),flush=True)
            if code!=0:
                # Existing authorized jobs continue naturally; no retries or new starts after failure.
                tasks.clear();state['status']='failed_no_automatic_retries';save()
                for pp,tt,ll in running.values():pp.wait();ll.close()
                raise RuntimeError('Worker failed; inspect '+str(task))
        for ds in c.PLAN['datasets']:
            if ds in scored:continue
            rr=[r for r in finished if r['dataset']==ds]
            if len(rr)==6 and all(r['exit_code']==0 for r in rr):
                with (c.OUT/(ds+'_summary.log')).open('w') as log:
                    subprocess.run([sys.executable,'-u',str(HERE/'summarize.py'),ds],stdout=log,stderr=subprocess.STDOUT,env=env(2),check=True)
                scored.append(ds);print('Fixed endpoint summary ready: '+ds,flush=True)
        save();time.sleep(10)
    c.verify_files();state['status']='completed_no_followups';save()
    print('Both benchmarks completed. No follow-up experiments launched.',flush=True)

if __name__=='__main__':main()
