"""Exactly nine repaired-L2 jobs; one active job on each authorized GPU1/6."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import common as c


def main():
    verification=json.loads((c.OUT/'verification.json').read_text())
    assert verification['passed'] and len(verification['three_seed_identity_checks'])==3
    assert not (c.OUT/'launcher_status.json').exists(), 'Do not launch twice'
    snapshot=c.OUT/'code_snapshot';snapshot.mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md'),
          c.ROOT/'experiments/pavia_transfer_audit_v1/verify.py',
          c.ROOT/'experiments/pavia_joint_v1/train.py',
          c.ROOT/'experiments/mluda_bida_extension_v1/common.py',
          c.ROOT/'experiments/relation_tokenizer_v1/relation.py',
          c.ROOT/'experiments/bn_stabilization_v1/normalization.py',
          c.ROOT/'experiments/bida_self_dropout/train.py',
          c.pavia.BIDA/'models/BiDA.py',c.pavia.BIDA/'utils/dataset.py',
          c.pavia.BIDA/'utils/utils_HSI.py',
          c.pavia.BIDA/'experiments/bida_memory_v1/run.py']
    hashes={}
    for file in files:
        shutil.copy2(str(file),str(snapshot/str(file).replace('/','__')))
        hashes[str(file)]=c.helper.file_hash(file)
    c.dump(c.OUT/'source_sha256.json',hashes)
    for command,name in ((['nvidia-smi'],'gpu_environment.txt'),
                          ([sys.executable,'-m','pip','freeze'],'packages.txt')):
        result=subprocess.run(command,capture_output=True,text=True)
        (c.OUT/name).write_text(result.stdout+result.stderr)
    link=c.ROOT/'results/pavia_l2_fixed_v1'
    assert not (link.exists() or link.is_symlink())
    link.symlink_to(c.OUT,target_is_directory=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    pending=[(arm,seed) for seed in (2100,2101,2102) for arm in ('A','B','C')]
    active={};completed=[]
    failed=False
    while pending or active:
        for gpu,(arm,seed,process,log) in list(active.items()):
            code=process.poll()
            if code is not None:
                completed.append({'arm':arm,'seed':seed,'gpu':gpu,'pid':process.pid,'exit_code':code})
                log.close();del active[gpu]
                if code!=0:failed=True
        if not failed:
            for gpu in (1,6):
                if gpu in active or not pending:continue
                assert all(c.helper.file_hash(Path(p))==v for p,v in hashes.items()), 'Source changed since snapshot'
                arm,seed=pending.pop(0)
                name='%s_%d'%(arm,seed)
                log=(c.OUT/(name+'.log')).open('x')
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('train.py')),
                     '--arm',arm,'--seed',str(seed),'--device',str(gpu),'--out',str(c.OUT/name)],
                     cwd=str(c.ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(arm,seed,process,log)
        c.dump(c.OUT/'launcher_status.json',{'active':[{'arm':a,'seed':s,'gpu':g,'pid':p.pid}
                   for g,(a,s,p,log) in active.items()], 'completed':completed,'pending':pending,
                   'failed':failed,'automatic_retries':False,'updated_at':time.time()})
        if failed and not active:
            raise RuntimeError('Run failed; queued tasks not launched, no automatic retry')
        if active:time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py'))],cwd=str(c.ROOT),check=True)


if __name__=='__main__':main()
