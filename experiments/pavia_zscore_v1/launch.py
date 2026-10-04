"""Exactly two seed2100 runs; no retries, no hyperparameter search."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import common as c


def main():
    assert json.loads((c.OUT/'implementation_checks.json').read_text())['all_checks_passed']
    if (c.OUT/'launcher_status.json').exists():raise RuntimeError('Do not launch twice')
    snapshot=c.OUT/'code_snapshot';snapshot.mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md'),
        c.ROOT/'experiments/pavia_joint_v1/train.py',c.ROOT/'experiments/mluda_bida_extension_v1/common.py',
        c.ROOT/'experiments/relation_tokenizer_v1/relation.py',
        c.ROOT/'experiments/bn_stabilization_v1/normalization.py',c.pavia.BIDA/'models/BiDA.py',
        c.pavia.BIDA/'utils/dataset.py',Path('/home/zhangzj26/TGRS_MLUDA-2024-main/utils.py')]
    hashes={}
    for p in files:
        shutil.copy2(str(p),str(snapshot/str(p).replace('/','__')));hashes[str(p)]=c.helper.file_hash(p)
    c.dump(c.OUT/'source_hashes.json',hashes)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    jobs=[]
    for arm,device in (('B',6),('C',7)):
        log=(c.OUT/(arm+'_2100.log')).open('x')
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('train.py')),
            '--arm',arm,'--seed','2100','--device',str(device),'--out',str(c.OUT/(arm+'_2100'))],
            cwd=str(c.ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((arm,device,process,log))
    while True:
        rows=[{'arm':arm,'seed':2100,'device':device,'pid':p.pid,'exit_code':p.poll()}
              for arm,device,p,log in jobs]
        c.dump(c.OUT/'launcher_status.json',{'jobs':rows,'no_automatic_retries':True})
        if all(row['exit_code'] is not None for row in rows):break
        time.sleep(15)
    for arm,device,p,log in jobs:log.close()
    if any(row['exit_code']!=0 for row in rows):raise RuntimeError('Diagnostic failed; inspect without changing algorithm')
    # Historical C may still be running; never stop or alter it.
    while not (c.reference('C')/'results.json').exists():
        log=c.ROOT/'results/mluda_bida_extension_v1/pavia_C_2100.log'
        if 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('Historical C incomplete')
        time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py'))],check=True)


if __name__=='__main__':main()
