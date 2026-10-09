import subprocess,sys,argparse,os
from common import OUT
p=argparse.ArgumentParser();p.add_argument('--seed',required=True);args=p.parse_args()
for arm in ['A','C']:
 with (OUT/f'{arm}_{args.seed}.log').open('x') as log:
  code=subprocess.call([sys.executable,'-u',str(OUT/'train.py'),'--arm',arm,'--seed',args.seed],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
 if code:sys.exit(code)
