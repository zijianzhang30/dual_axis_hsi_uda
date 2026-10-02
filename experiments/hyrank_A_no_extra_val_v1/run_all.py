"""Launch exactly three seeds once, leave old experiments read-only."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from control import OUT,ROOT,SEEDS,c,dump


def main():
    assert shutil.disk_usage(OUT).free>5*1024**3
    assert not (OUT/'launcher_status.json').exists(), 'Refuse duplicate launch'
    assert (OUT/'preflight.json').exists()
    here=Path(__file__).resolve().parent
    snapshot=OUT/'code_snapshot';snapshot.mkdir()
    for path in here.glob('*'):
        if path.is_file():shutil.copy2(path,snapshot/path.name)
    dump(OUT/'protocol.json',{'seeds':list(SEEDS),'epochs':200,'steps_per_epoch':80,'arm':'A_no_extra_val',
         'interpretation':'remove extra epoch1 source-val workflow control, not pure RNG',
         'source_val_calls':0,'target_eval_epochs':list(range(10,201,10)),
         'target_eval':'original released call position after scheduler; original function; no target-best official selection',
         'mode':'no new train/eval call; native train() at epoch start preserved',
         'rng':'one original shared loader generator; original startup seed only; no restore, independent generator or artificial consumption',
         'audit':'actual already-consumed dataset indices and RNG states; no loader replay',
         'new_vs_old':'Cancel extra epoch1 source validation. Requested native target evaluation occupies every10 evaluation position instead of old source validation. This distinction is disclosed, not simulated.',
         'output':'new NAS directory only; do not replace old A/B/C tables or launch further tasks'})
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',
             TMPDIR=tempfile.mkdtemp(prefix='hyrank_A_no_extra_val_'))
    jobs=[]
    for seed in SEEDS:
        log=(OUT/f'seed_{seed}.log').open('x')
        p=subprocess.Popen([sys.executable,'-u',str(here/'train.py'),'--seed',str(seed),'--device','6'],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((seed,p,log))
    def status():
        dump(OUT/'launcher_status.json',{'jobs':[{'seed':seed,'device':6,'pid':p.pid,'exit_code':p.poll()}
                                               for seed,p,_ in jobs]})
    status()
    while any(p.poll() is None for _,p,_ in jobs):time.sleep(15);status()
    for _,_,log in jobs:log.close()
    assert all(p.returncode==0 for _,p,_ in jobs), 'Failure; no automatic rerun or extra tasks'
    subprocess.run([sys.executable,'-u',str(here/'summarize.py')],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
