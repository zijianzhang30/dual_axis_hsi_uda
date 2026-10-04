"""Start a durable launcher once, only after GPU and NAS guards pass."""
import subprocess
import sys
from pathlib import Path
from control import OUT,ROOT


def main():
    assert not (OUT/'launcher_status.json').exists(),'Refuse duplicate launch'
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(s.split(',')[0]):int(s.split(',')[1]) for s in query.splitlines()}
    assert memory[6]<1000, 'GPU6 is no longer idle; do not start or choose another GPU automatically'
    with (OUT/'launcher.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('run_all.py'))],
                           cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True)
    print('HyRANK A_no_extra_val launcher PID:',p.pid,flush=True)


if __name__=='__main__':main()
