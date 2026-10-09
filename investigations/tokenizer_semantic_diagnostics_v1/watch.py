import pathlib,json,time,os,subprocess
O=pathlib.Path(__file__).parent;launch=json.loads((O/'launch.json').read_text())
while True:
 done=[];active=[];failed=[]
 for row in launch['active']:
  s=row['seed']
  if (O/f'seed_{s}/summary.json').exists():done.append(s)
  elif pathlib.Path(f"/proc/{row['pid']}").exists():active.append(row)
  else:failed.append(row)
 (O/'status.json').write_text(json.dumps({'state':'diagnosing' if not failed else 'failed','completed_seeds':done,'active':active,'failed':failed,'no_training':True,'no_auto_retry':True,'updated':time.time()},indent=2)+'\n')
 if failed:break
 if len(done)==3:
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  with (O/'report.log').open('x') as f:code=subprocess.call(['/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python','-u',str(O/'report.py')],cwd=O,env=env,stdout=f,stderr=subprocess.STDOUT)
  if code:(O/'status.json').write_text(json.dumps({'state':'report_failed','exit_code':code})+'\n')
  break
 time.sleep(15)
