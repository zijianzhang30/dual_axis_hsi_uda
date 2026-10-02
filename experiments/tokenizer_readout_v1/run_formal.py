"""Run the new stress-seed cells; completed control is not rerun."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',required=True)
    parser.add_argument('--arms',nargs='+',choices=('bida_class','query_mean','bida_mean'),required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=root/'results/tokenizer_readout_v1';out.mkdir(parents=True,exist_ok=True)
    for arm in args.arms:
        folder=out/f'{arm}_2101'
        if (folder/'results.json').exists():
            print(json.dumps({'complete_existing':str(folder)}),flush=True);continue
        if folder.exists() and any(folder.iterdir()):raise RuntimeError(f'Incomplete run requires inspection: {folder}')
        command=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--arm',arm,'--seed','2101','--device',args.device,'--out',str(folder)]
        print(json.dumps({'starting':arm,'device':args.device}),flush=True)
        with (out/f'{arm}_2101.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait()!=0:raise RuntimeError(f'Run failed: {folder}')

if __name__=='__main__':main()
