"""Preserve sources and logs; run nine jobs, then score the frozen predictions."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/relation_tokenizer_v1'


def main():
    checks=json.loads((OUT/'implementation_checks.json').read_text())
    assert all(x['all_checks_passed'] for x in checks['runs'])
    snapshot=OUT/'code_snapshot'
    snapshot.mkdir()
    directory=Path(__file__).parent
    paths=[*directory.glob('*.py'),directory/'PROTOCOL.md',
           ROOT/'experiments/bn_stabilization_v1/normalization.py',
           Path('/home/zhangzj26/IEEE_TCSVT_BiDA/models/BiDA.py'),
           Path('/home/zhangzj26/IEEE_TCSVT_BiDA/utils/dataset.py'),
           Path('/home/zhangzj26/IEEE_TCSVT_BiDA/utils/scheduler.py'),
           Path('/home/zhangzj26/IEEE_TCSVT_BiDA/utils/utils_HSI.py'),
           Path('/home/zhangzj26/IEEE_TCSVT_BiDA/experiments/bida_memory_v1/run.py'),
           ROOT/'experiments/bida_self_dropout/train.py']
    for i,path in enumerate(paths):shutil.copyfile(path,snapshot/f'{i:02d}_{path.name}')
    diffs=[]
    for path in directory.glob('*.py'):
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path)],capture_output=True,text=True)
        assert result.returncode in (0,1)
        diffs.append(result.stdout)
    (OUT/'CODE_DIFF.patch').write_text('\n'.join(diffs))
    jobs=[];handles=[]
    for i,(arm,seed) in enumerate((a,s) for a in ('A','B','C') for s in (2100,2101,2102)):
        gpu=(4,5,6,7)[i%4]
        name=f'{arm}_{seed}'
        log=(OUT/f'{name}.log').open('x');handles.append(log)
        command=[sys.executable,'-u',str(directory/'train.py'),'--arm',arm,'--seed',str(seed),
                 '--device',str(gpu),'--out',str(OUT/name)]
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,proc,gpu))
    status={n:{'pid':p.pid,'gpu':g,'state':'running'} for n,p,g in jobs}
    (OUT/'runner_status.json').write_text(json.dumps(status,indent=2)+'\n')
    for name,proc,_ in jobs:
        status[name].update(state='finished',exit_code=proc.wait())
        (OUT/'runner_status.json').write_text(json.dumps(status,indent=2)+'\n')
    for handle in handles:handle.close()
    if any(x['exit_code']!=0 for x in status.values()):
        raise RuntimeError('A run failed; inspect logs. Do not score incomplete arms.')
    report=subprocess.run([sys.executable,str(directory/'summarize.py')],cwd=ROOT,
                          capture_output=True,text=True)
    if report.returncode:
        print(report.stdout,flush=True);print(report.stderr,flush=True)
        raise RuntimeError('Summary audit failed')
    (directory/'RESULTS.md').write_text(report.stdout)
    print(report.stdout,flush=True)


if __name__=='__main__':main()
