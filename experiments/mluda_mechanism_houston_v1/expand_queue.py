"""Scheduling-only takeover: adopt current tasks unchanged, expand to GPUs0/1/6/7."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import OUT, SEEDS, ARMS, dump

GPUS=(0,1,2,3)

def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists(): return False
    try: return p.read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError: return False

def control(state):
    pending=state['pending']; completed=state['completed']
    active={r['gpu']:r for r in state['active']}
    handles={}
    while pending or active:
        for gpu,r in list(active.items()):
            if alive(r['pid']): continue
            folder=OUT/('%s_%d'%(r['arm'],r['seed']))
            frozen=folder/'frozen.json'; history=folder/'history.jsonl'
            rows=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
            if not frozen.exists() or len(rows)!=100 or rows[-1]['updates']!=3800:
                dump(OUT/'launcher_status.json',{'state':'failed','task':r,'completed':completed,'pending':pending,'active':list(active.values()),'updated':time.time()})
                raise RuntimeError('Task failed; no retry or additional experiments')
            completed.append([r['arm'],r['seed']]); del active[gpu]
            if r['pid'] in handles:
                p,l=handles.pop(r['pid']); p.wait(); l.close()
        for gpu in GPUS:
            if gpu in active or not pending: continue
            arm,seed=pending.pop(0)
            assert not (OUT/('%s_%d'%(arm,seed))).exists()
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
            log=(OUT/('%s_%d.log'%(arm,seed))).open('x')
            p=subprocess.Popen([sys.executable,'-u',str(OUT/'run.py'),'--arm',arm,'--seed',str(seed)],cwd=str(OUT),env=env,stdout=log,stderr=subprocess.STDOUT)
            handles[p.pid]=(p,log)
            active[gpu]={'gpu':gpu,'pid':p.pid,'arm':arm,'seed':seed}
        dump(OUT/'launcher_status.json',{'state':'running','pending':pending,'completed':completed,'active':list(active.values()),'physical_gpus':GPUS,'scheduling_only_expansion':True,'updated':time.time()})
        if active: time.sleep(15)
    with (OUT/'summary.log').open('x') as log:
        code=subprocess.call([sys.executable,'-u',str(OUT/'summarize.py')],cwd=str(OUT),stdout=log,stderr=subprocess.STDOUT)
    dump(OUT/'launcher_status.json',{'state':'completed' if code==0 else 'audit_or_scoring_failed','completed':completed,'summary_exit_code':code,'updated':time.time(),'physical_gpus':GPUS})
    assert code==0

def takeover():
    launch=json.loads((OUT/'launch.json').read_text()); pid=launch['pid']
    assert alive(pid)
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
    assert str(OUT/'launch.py').encode() in cmd
    os.kill(pid,signal.SIGSTOP)
    try:
        time.sleep(.5)
        state=json.loads((OUT/'launcher_status.json').read_text())
        assert state['state']=='running'
        expected={(r['arm'],r['seed']) for r in state['active']}
        actual=set()
        for p in Path('/proc').iterdir():
            if not p.name.isdigit(): continue
            try: parts=(p/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError): continue
            if str(OUT/'run.py').encode() in parts:
                args=[x.decode() for x in parts if x]
                actual.add((args[args.index('--arm')+1],int(args[args.index('--seed')+1])))
        assert actual==expected,(actual,expected)
        all_tasks={(a,s) for s in SEEDS for a in ARMS}
        assert all_tasks==expected|{tuple(x) for x in state['pending']}|{tuple(x) for x in state['completed']}
        dump(OUT/'concurrency_expansion.json',{'requested':'user approved 4 or 6 GPUs','selected':GPUS,'reason':'GPU0/1 spare; GPU2-5 already busy','previous_launcher_pid':pid,'preserved_state':state,'time':time.time(),'training_scripts_changed':False})
        os.kill(pid,signal.SIGTERM)
        os.kill(pid,signal.SIGCONT)
        state_path=OUT/'adopted_queue_state.json'; dump(state_path,state)
        with (OUT/'expanded_launcher.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-u',str(OUT/'expand_queue.py'),'--control'],cwd=str(OUT),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(OUT/'expanded_launch.json',{'pid':p.pid,'physical_gpus':GPUS,'started_at':time.time()})
        print('Expanded controller PID',p.pid)
    except Exception:
        if alive(pid): os.kill(pid,signal.SIGCONT)
        raise

if __name__=='__main__':
    if '--control' in sys.argv: control(json.loads((OUT/'adopted_queue_state.json').read_text()))
    else: takeover()
