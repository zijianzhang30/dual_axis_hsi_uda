"""Resume inference only after disk-full; preserve all training and past outputs."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/relation_bn_2x2_v1'
HERE=Path(__file__).parent


def main():
    assert OUT.is_symlink()
    assert all((OUT/f'cell2_{s}/training_manifest.json').exists() for s in (2100,2101,2102))
    env=dict(os.environ)
    env['TMPDIR']=tempfile.mkdtemp(prefix='rbn_',dir='/dev/shm')
    shutil.copyfile(HERE/'grid.py',OUT/'code_snapshot/grid_resumed.py')
    shutil.copyfile(Path(__file__),OUT/'code_snapshot/recover_grid.py')
    status={};jobs=[];logs=[]
    for cell,gpu in zip(('1','2','4'),(6,7,6)):
        log=(OUT/f'grid_cell{cell}_recovery.log').open('x');logs.append(log)
        p=subprocess.Popen([sys.executable,'-u',str(HERE/'grid.py'),'--cell',cell,'--device',str(gpu),'--resume'],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((cell,p));status[cell]={'pid':p.pid,'type':'inference resume only','state':'running'}
    path=OUT/'recovery_status.json';path.write_text(json.dumps(status,indent=2)+'\n')
    for cell,p in jobs:
        status[cell].update(state='finished',exit_code=p.wait());path.write_text(json.dumps(status,indent=2)+'\n')
    for log in logs:log.close()
    if any(x['exit_code']!=0 for x in status.values()):raise RuntimeError('Inference recovery failed; do not score')
    report=subprocess.run([sys.executable,str(HERE/'summarize.py')],cwd=ROOT,env=env,capture_output=True,text=True)
    if report.returncode:print(report.stdout+report.stderr,flush=True);raise RuntimeError('Summary audit failed')
    (HERE/'RESULTS.md').write_text(report.stdout)
    print(report.stdout,flush=True)


if __name__=='__main__':main()
