"""Track existing Pavia processes only; never start or restart training."""
import json
from pathlib import Path
import subprocess
import sys
import time
import common as c


def main():
    jobs=((2100,6,555186),(2101,6,555188),(2102,7,555191))
    while True:
        rows=[]
        for seed,device,pid in jobs:
            folder=c.OUT/'pavia'/('C_%d'%seed)
            lines=(folder/'history.jsonl').read_text().splitlines()
            try:epoch=json.loads(lines[-1])['epoch'] if lines else 0
            except json.JSONDecodeError:epoch=0
            complete=(folder/'results.json').exists()
            active=Path('/proc/%d'%pid).exists()
            rows.append({'seed':seed,'device':device,'pid':pid,'epoch':epoch,
                         'complete':complete,'process_exists':active})
        c.dump(c.OUT/'pavia_only_status.json',{'pavia':rows,
                'shanghai':'stopped/deferred at explicit user request; preserve incomplete files',
                'automatic_training_launches':False})
        if all(row['complete'] for row in rows):break
        if any(not row['process_exists'] and not row['complete'] for row in rows):
            raise RuntimeError('Pavia exited without final results; no automatic restart')
        time.sleep(15)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),
                    '--tasks','pavia'],check=True)


if __name__=='__main__':main()
