import concurrent.futures,json,os,subprocess,sys,time
import study as s

def worker(seed):
 for arm in s.ARMS:
  mf=s.OUT/f'{arm}_{seed}'/'FINAL_PREDICTIONS_FROZEN.json'
  if mf.exists():
   record=json.loads(mf.read_text());assert record['completed_epochs']==200 and not record['target_GT_scored']
   for rel,h in record['files_sha256'].items():assert s.sha(mf.parent/rel)==h
   print('REUSE_COMPLETED_NO_RETRAIN',arm,seed,flush=True);continue
  path=s.OUT/'logs'/f'{arm}_{seed}.log'
  with path.open('x') as log:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
   proc=subprocess.run([sys.executable,str(s.HERE/'train.py'),'--arm',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(s.ROOT))
  if proc.returncode:raise RuntimeError(f'{arm}_{seed} failed code{proc.returncode}; see{path}; no automatic retry')
  print('RUN_FINISHED',arm,seed,flush=True)
 return seed

def main():
 s.verify();start=time.monotonic()
 # Three authorized seeds in parallel, no subagents, GPU2 ample memory, other jobs untouched.
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(worker,seed) for seed in s.SEEDS]
  for f in concurrent.futures.as_completed(futures):f.result()
 records={}
 for seed in s.SEEDS:
  for arm in s.ARMS:
   folder=s.OUT/f'{arm}_{seed}';mf=json.loads((folder/'FINAL_PREDICTIONS_FROZEN.json').read_text());assert mf['completed_epochs']==200 and not mf['target_GT_scored']
   for rel,h in mf['files_sha256'].items():assert s.sha(folder/rel)==h;records[str(folder/rel)]=h
   records[str(folder/'FINAL_PREDICTIONS_FROZEN.json')]=s.sha(folder/'FINAL_PREDICTIONS_FROZEN.json')
 s.dump(s.OUT/'ALL_NINE_FINAL_PREDICTIONS_FROZEN.json',{'target_GT_scored':False,'all9_complete':True,'elapsed_seconds':time.monotonic()-start,'files_sha256':records})
 result=subprocess.run([sys.executable,str(s.HERE/'score.py')],cwd=str(s.ROOT));assert result.returncode==0
if __name__=='__main__':main()
