"""Two shared-GPU read-only workers, exactly three seed analyses, no retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import common as t


def main():
    assert not (t.OUT/'launcher_status.json').exists()
    inventory=json.loads((t.OUT/'inventory.json').read_text())
    assert inventory['only_repaired_L2_runs']
    pending=list(t.SEEDS);active={};completed=[];failed=False
    while pending or active:
        for gpu,(seed,p,log) in list(active.items()):
            code=p.poll()
            if code is not None:
                completed.append({'seed':seed,'physical_gpu':gpu,'pid':p.pid,'exit_code':code})
                log.close();del active[gpu]
                if code!=0:failed=True
        if not failed:
            for gpu in (2,3):
                if gpu in active or not pending:continue
                assert all(t.c.helper.file_hash(Path(path))==value for path,value in inventory['source_sha256'].items())
                seed=pending.pop(0)
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                log=(t.OUT/('seed_%d.log'%seed)).open('x')
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('worker.py')),
                           '--seed',str(seed),'--device','0'],cwd=str(t.ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(seed,process,log)
        t.c.dump(t.OUT/'launcher_status.json',{'active':[{'seed':seed,'physical_gpu':gpu,'pid':p.pid}
                     for gpu,(seed,p,log) in active.items()], 'completed':completed,'pending':pending,
                     'failed':failed,'shared_GPU_inference_authorized_by_user':True,
                     'training_tasks_started':0,'automatic_retries':False,'updated_at':time.time()})
        if failed and not active:raise RuntimeError('Read-only analysis failed; report, do not auto-retry')
        if active:time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py'))],cwd=str(t.ROOT),check=True)


if __name__=='__main__':main()
