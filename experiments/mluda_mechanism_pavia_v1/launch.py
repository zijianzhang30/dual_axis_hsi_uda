"""Fixed twelve-task queue; no retry/tuning/additional jobs. GPU6/7, one each."""
import json
import os
import subprocess
import sys
import time

from common import OUT, ARMS, SEEDS, dump


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['passed']
    assert not (OUT/'launcher_status.json').exists()
    pending=[(arm,seed) for seed in SEEDS for arm in ARMS]
    active={}; completed=[]
    while pending or active:
        for gpu in (6,7):
            if gpu not in active and pending:
                arm,seed=pending.pop(0)
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
                log=(OUT/('%s_%d.log'%(arm,seed))).open('x')
                proc=subprocess.Popen([sys.executable,'-u',str(OUT/'run.py'),'--arm',arm,'--seed',str(seed)],
                    cwd=str(OUT),env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(proc,arm,seed,log)
        dump(OUT/'launcher_status.json',{'state':'running','pending':pending,
            'active':[{'gpu':g,'pid':p.pid,'arm':a,'seed':s} for g,(p,a,s,l) in active.items()],
            'completed':completed,'updated':time.time()})
        time.sleep(15)
        for gpu,(proc,arm,seed,log) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            log.close(); del active[gpu]
            if code!=0:
                # Allow already running independent task to finish; do not start more.
                dump(OUT/'launcher_status.json',{'state':'failed','arm':arm,'seed':seed,'exit_code':code,
                    'completed':completed,'pending':pending,'remaining_active':[p.pid for p,a,s,l in active.values()]})
                raise RuntimeError('Training failed; inspect log. No automatic retry.')
            completed.append((arm,seed))
    with (OUT/'summary.log').open('x') as log:
        code=subprocess.call([sys.executable,'-u',str(OUT/'summarize.py')],cwd=str(OUT),stdout=log,stderr=subprocess.STDOUT)
    dump(OUT/'launcher_status.json',{'state':'completed' if code==0 else 'audit_or_scoring_failed',
         'completed':completed,'summary_exit_code':code,'updated':time.time()})
    assert code==0


if __name__=='__main__':
    main()
