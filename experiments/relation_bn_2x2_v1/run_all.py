"""Exactly three new training tasks, then four independent grid inference workers."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/relation_bn_2x2_v1'
HERE=Path(__file__).parent


def main():
    assert json.loads((OUT/'reuse_audit.json').read_text())['eligible_for_reuse']
    assert all(x['all_checks_passed'] for x in json.loads((OUT/'implementation_checks.json').read_text()))
    snapshot=OUT/'code_snapshot';snapshot.mkdir()
    paths=[*HERE.glob('*.py'),HERE/'PROTOCOL.md',
           ROOT/'experiments/relation_tokenizer_v1/relation.py',ROOT/'experiments/relation_tokenizer_v1/train.py',
           Path('/home/zhangzj26/IEEE_TCSVT_BiDA/models/BiDA.py'),Path('/home/zhangzj26/IEEE_TCSVT_BiDA/utils/dataset.py'),
           ROOT/'experiments/bn_stabilization_v1/normalization.py']
    for i,path in enumerate(paths):shutil.copyfile(path,snapshot/f'{i:02d}_{path.name}')
    diffs=[]
    for path in HERE.glob('*.py'):
        r=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path)],capture_output=True,text=True)
        assert r.returncode in (0,1);diffs.append(r.stdout)
    (OUT/'CODE_DIFF.patch').write_text('\n'.join(diffs))
    jobs=[];logs=[];status={}
    for seed,gpu in zip((2100,2101,2102),(6,7,6)):
        name=f'cell2_{seed}';log=(OUT/f'{name}.log').open('x');logs.append(log)
        p=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--seed',str(seed),'--device',str(gpu),'--out',str(OUT/name)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,p));status[name]={'pid':p.pid,'gpu':gpu,'type':'training','state':'running'}
    status_path=OUT/'runner_status.json';status_path.write_text(json.dumps(status,indent=2)+'\n')
    for name,p in jobs:
        status[name].update(state='finished',exit_code=p.wait());status_path.write_text(json.dumps(status,indent=2)+'\n')
    if any(x['exit_code']!=0 for x in status.values()):raise RuntimeError('Training failed; stop without extra tasks')
    grid_jobs=[]
    for cell,gpu in zip(('1','2','3','4'),(6,7,6,7)):
        name=f'grid_cell{cell}';log=(OUT/f'{name}.log').open('x');logs.append(log)
        p=subprocess.Popen([sys.executable,'-u',str(HERE/'grid.py'),'--cell',cell,'--device',str(gpu)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        grid_jobs.append((name,p));status[name]={'pid':p.pid,'gpu':gpu,'type':'inference only','state':'running'}
    status_path.write_text(json.dumps(status,indent=2)+'\n')
    for name,p in grid_jobs:
        status[name].update(state='finished',exit_code=p.wait());status_path.write_text(json.dumps(status,indent=2)+'\n')
    for log in logs:log.close()
    if any(x['exit_code']!=0 for x in status.values()):raise RuntimeError('Grid failed; do not score incomplete predictions')
    r=subprocess.run([sys.executable,str(HERE/'summarize.py')],cwd=ROOT,capture_output=True,text=True)
    if r.returncode:print(r.stdout+r.stderr,flush=True);raise RuntimeError('Summary audit failed')
    (HERE/'RESULTS.md').write_text(r.stdout);print(r.stdout,flush=True)


if __name__=='__main__':main()
