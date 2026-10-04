"""Adopt live training unchanged and expand only the pending queue to GPUs1/2/3/6."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import common as c

OLD_DISPATCHER=998986
GPUS=(1,6,2,3)


def process(pid):
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        return {'state':stat[0],'ppid':int(stat[1]),'cmd':(path/'cmdline').read_bytes()}
    except FileNotFoundError:return None


def validate_sources(hashes):
    assert all(c.helper.file_hash(Path(path))==expected for path,expected in hashes.items())


def completed_adopted(row):
    folder=c.OUT/('%s_%d'%(row['arm'],row['seed']))
    history=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
    assert len(history)==200 and history[-1]['epoch']==200
    result=json.loads((folder/'results.json').read_text())
    assert result['frozen_manifest_sha256']==c.helper.file_hash(folder/'frozen_manifest.json')
    assert result['config']['arm']==row['arm'] and result['config']['seed']==row['seed']


def main():
    assert not (c.OUT/'queue_expansion.json').exists(), 'Do not take over twice'
    hashes=json.loads((c.OUT/'source_sha256.json').read_text())
    validate_sources(hashes)
    old=process(OLD_DISPATCHER)
    assert old and b'pavia_l2_fixed_v1/launch.py' in old['cmd']
    os.kill(OLD_DISPATCHER,signal.SIGSTOP)
    stopped=True
    try:
        status=json.loads((c.OUT/'launcher_status.json').read_text())
        assert not status['failed']
        active_rows=status['active']
        children=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            record=process(int(path.name))
            if record and record['ppid']==OLD_DISPATCHER and b'pavia_l2_fixed_v1/train.py' in record['cmd']:
                children.append(int(path.name))
        assert set(children)=={r['pid'] for r in active_rows}, 'Queue changed at takeover; leave old scheduler intact'
        for row in active_rows:
            state=process(row['pid'])
            assert state and b'pavia_l2_fixed_v1/train.py' in state['cmd']
            assert ('--arm\x00'+row['arm']).encode() in state['cmd']
        planned={(arm,seed) for arm in ('A','B','C') for seed in (2100,2101,2102)}
        known=[(r['arm'],r['seed']) for r in active_rows+status['completed']]+[tuple(r) for r in status['pending']]
        assert len(known)==9 and set(known)==planned
        c.dump(c.OUT/'launcher_status_before_expansion.json',status)
        c.dump(c.OUT/'queue_expansion.json',{'old_dispatcher_pid':OLD_DISPATCHER,
               'new_dispatcher_pid':os.getpid(),'gpus':GPUS,'training_pids_unchanged':[r['pid'] for r in active_rows],
               'pending_task_count':len(status['pending']),'code_sha256':c.helper.file_hash(Path(__file__)),
               'time':time.time(),'change':'scheduler only; no training restart, RNG/model/config change'})
        # Terminate only the old scheduling process, not its process group or children.
        os.kill(OLD_DISPATCHER,signal.SIGTERM)
        os.kill(OLD_DISPATCHER,signal.SIGCONT)
        stopped=False
        while process(OLD_DISPATCHER) and process(OLD_DISPATCHER)['state']!='Z':time.sleep(.1)
    finally:
        if stopped:os.kill(OLD_DISPATCHER,signal.SIGCONT)
    active={row['gpu']:{'row':row,'process':None,'log':None} for row in active_rows}
    pending=[tuple(row) for row in status['pending']]
    finished=status['completed'];failed=False
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    while pending or active:
        for gpu,job in list(active.items()):
            child=job['process'];row=job['row']
            if child is None:
                observed=process(row['pid'])
                if observed and observed['state']!='Z':continue
                try:
                    completed_adopted(row)
                    code=0;method='frozen result and full history verified; adopted child wait status unavailable'
                except Exception as error:
                    code=1;method=repr(error)
            else:
                code=child.poll()
                if code is None:continue
                method='child process exit code'
                job['log'].close()
            finished.append(dict(row,exit_code=code,completion_check=method))
            del active[gpu]
            if code!=0:failed=True
        if not failed:
            for gpu in GPUS:
                if gpu in active or not pending:continue
                validate_sources(hashes)
                arm,seed=pending.pop(0);name='%s_%d'%(arm,seed)
                assert not (c.OUT/name).exists(), 'Never duplicate or overwrite a run'
                log=(c.OUT/(name+'.log')).open('x')
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('train.py')),
                     '--arm',arm,'--seed',str(seed),'--device',str(gpu),'--out',str(c.OUT/name)],
                     cwd=str(c.ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]={'row':{'arm':arm,'seed':seed,'gpu':gpu,'pid':child.pid},'process':child,'log':log}
        c.dump(c.OUT/'launcher_status.json',{'active':[job['row'] for job in active.values()],
                   'completed':finished,'pending':pending,'failed':failed,'automatic_retries':False,
                   'authorized_gpus':GPUS,'dispatcher_pid':os.getpid(),'updated_at':time.time()})
        if failed and not active:raise RuntimeError('Training failed; no retry or new queued tasks')
        if active:time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py'))],cwd=str(c.ROOT),check=True)


if __name__=='__main__':main()
