"""Non-invasive idle-GPU queue; never stop other processes or retry tasks."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_norm_sampling_v1')
HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus',default='1,2,3,6')
    parser.add_argument('--share',action='store_true',help='Only after explicit full-training sharing authorization')
    args=parser.parse_args(); gpus=[int(x) for x in args.gpus.split(',')]
    assert json.loads((OUT/'verification.json').read_text())['passed']
    jobs=[(a,s) for s in (2100,2101,2102) for a in ('P0','P1','P2','P3')]
    assert not (OUT/'queue_status.json').exists(), 'Do not start duplicate queue'
    active={}; done=[]; failed=[]
    while jobs or active:
        for gpu,(proc,arm,seed,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close(); (done if code==0 else failed).append({'arm':arm,'seed':seed,'exit_code':code})
                del active[gpu]
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        state={int(x.split(',')[0]):tuple(int(v.strip()) for v in x.split(',')[1:]) for x in raw.splitlines()}
        for gpu in gpus:
            memory,util=state[gpu]
            if jobs and gpu not in active and ((memory<1500 and util<10) or (args.share and memory<74000)):
                arm,seed=jobs.pop(0)
                log=(OUT/('%s_%d.log'%(arm,seed))).open('w')
                proc=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--arm',arm,'--seed',str(seed),'--device',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(proc,arm,seed,log)
        status={'pending':[{'arm':a,'seed':s} for a,s in jobs],
                'active':[{'gpu':g,'pid':p.pid,'arm':a,'seed':s} for g,(p,a,s,l) in active.items()],
                'completed':done,'failed':failed,'share':args.share,'gpus':gpus,'updated_unix':time.time()}
        (OUT/'queue_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if jobs or active: time.sleep(30)
    if not failed:
        subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)


if __name__=='__main__':main()
