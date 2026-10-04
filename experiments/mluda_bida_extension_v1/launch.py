"""Run exactly nine prescribed tasks on GPUs 6/7, then summarize once."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import common as c


def main():
    checks=__import__('json').loads((c.OUT/'implementation_checks.json').read_text())
    assert checks['all_checks_passed']
    if (c.OUT/'launcher_status.json').exists(): raise RuntimeError('Do not launch twice')
    snapshot=c.OUT/'code_snapshot';snapshot.mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md'),
        c.ROOT/'experiments/pavia_joint_v1/train.py',c.ROOT/'experiments/pavia_joint_v1/PROTOCOL.md',
        c.ROOT/'experiments/relation_tokenizer_v1/relation.py',
        c.ROOT/'experiments/bn_stabilization_v1/normalization.py',
        c.pavia.BIDA/'models/BiDA.py',c.pavia.BIDA/'utils/dataset.py',
        c.ROOT/'experiments/bida_self_dropout/train.py']
    hashes={}
    for path in files:
        destination=snapshot/(str(path).replace('/','__'))
        shutil.copy2(str(path),str(destination));hashes[str(path)]=c.helper.file_hash(path)
    c.dump(c.OUT/'source_hashes.json',hashes)
    queue=[('pavia','C',seed) for seed in (2100,2101,2102)]
    queue += [('shanghai',arm,seed) for seed in (2100,2101,2102) for arm in ('B','C')]
    active=[];finished=[]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    while queue or active:
        for job in active[:]:
            code=job['process'].poll()
            if code is not None:
                job['log'].close();finished.append({k:v for k,v in job.items() if k not in ('process','log')})
                finished[-1]['exit_code']=code;active.remove(job)
        for device in (6,7):
            while queue and sum(j['device']==device for j in active)<2:
                if any(j['exit_code']!=0 for j in finished):break
                task,arm,seed=queue.pop(0);out=c.OUT/task/('%s_%d'%(arm,seed))
                log_path=c.OUT/('%s_%s_%d.log'%(task,arm,seed));log=log_path.open('w')
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),
                     '--task',task,'--arm',arm,'--seed',str(seed),'--device',str(device),'--out',str(out)]
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,
                                         cwd=str(c.ROOT))
                active.append({'task':task,'arm':arm,'seed':seed,'device':device,'pid':process.pid,
                               'process':process,'log':log,'exit_code':None})
        c.dump(c.OUT/'launcher_status.json',{'finished':finished,'running':[
            {k:v for k,v in j.items() if k not in ('process','log')} for j in active],
            'pending':queue,'no_automatic_retries':True})
        if not active and any(j['exit_code']!=0 for j in finished):
            raise RuntimeError('A prescribed task failed; inspect without changing method or adding runs')
        if active:time.sleep(15)
    assert len(finished)==9 and all(j['exit_code']==0 for j in finished)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py'))],check=True)


if __name__=='__main__':main()
