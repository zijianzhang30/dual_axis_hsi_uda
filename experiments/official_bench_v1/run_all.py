"""Durable, bounded HyRANK execution; never touches other experiments."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/official_bench_v1'
HERE=Path(__file__).resolve().parent


def main():
    assert shutil.disk_usage(OUT).free>10*1024**3
    assert json.loads((OUT/'implementation_checks.json').read_text())['all_passed']
    assert not (OUT/'launcher_status.json').exists(), 'Refuse duplicate launch'
    snapshot=OUT/'code_snapshot';snapshot.mkdir()
    for path in HERE.iterdir():
        if path.suffix in ('.py','.md'):shutil.copy2(path,snapshot/path.name)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             TMPDIR=tempfile.mkdtemp(prefix='bida_hyrank_',dir='/dev/shm'),PYTHONUNBUFFERED='1')
    running=[]
    for index,(seed,arm) in enumerate((s,a) for s in (2100,2101,2102) for a in 'ABC'):
        gpu=6+index%2
        log=(OUT/f'hyrank_{arm}_{seed}.log').open('x')
        process=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--arm',arm,
                                  '--seed',str(seed),'--device',str(gpu)],
                                 cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        running.append({'process':process,'log':log,'arm':arm,'seed':seed,'gpu':gpu})
    def status():
        values=[{'arm':r['arm'],'seed':r['seed'],'gpu':r['gpu'],'pid':r['process'].pid,
                 'exit_code':r['process'].poll()} for r in running]
        (OUT/'launcher_status.json').write_text(json.dumps({'jobs':values},indent=2)+'\n')
    status()
    while any(r['process'].poll() is None for r in running):
        time.sleep(15);status()
    for r in running:r['log'].close()
    if any(r['process'].returncode!=0 for r in running):
        raise RuntimeError('Training failure; no automatic restart or target evaluation')
    subprocess.run([sys.executable,str(HERE/'summarize.py')],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
