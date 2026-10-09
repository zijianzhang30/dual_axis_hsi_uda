"""Only 15 repaired-normalized Pavia runs. No automatic retry or extra dataset."""
import os,sys,time,json,subprocess
from common import OUT,SEEDS,ARMS,dump,sha
GPUS=[0,4,5]
def main():
 assert json.loads((OUT/'verification.json').read_text())['passed']
 # Respect existing raw-input G: wait for its controller to finish naturally.
 old=OUT.parent/'tokenizer_capacity_g_v1/status.json'
 while True:
  s=json.loads(old.read_text())
  if s['state']=='completed':break
  if 'failed' in s['state'] or 'error' in s['state']:
   dump(OUT/'status.json',{'state':'blocked_by_existing_G_failure','old_status':s,'no_auto_retry':True});return
  dump(OUT/'status.json',{'state':'waiting_for_existing_raw_G_to_finish_naturally','old_state':s['state'],'normalized_training_started':False,'updated':time.time()});time.sleep(20)
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 completed=[]
 # A establishes exact normalized-data training/RNG hashes, one control per seed.
 for stage,pending in [('A_controls',[('A',s) for s in SEEDS]),('mechanism',[ (a,s) for s in SEEDS for a in 'BCFG'])]:
  active={}
  while pending or active:
   for gpu,(p,a,s,l) in list(active.items()):
    code=p.poll()
    if code is None:continue
    l.close();del active[gpu]
    if code:
     dump(OUT/'status.json',{'state':'failed','arm':a,'seed':s,'exit_code':code,'remaining_running':[q.pid for q,_,_,_ in active.values()],'no_automatic_retry':True});return
    completed.append([a,s])
   for gpu in GPUS:
    if gpu in active or not pending:continue
    a,s=pending.pop(0);e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu)
    l=(OUT/f'{a}_{s}.log').open('x');p=subprocess.Popen([sys.executable,'-u',str(OUT/'train.py'),'--dataset','pavia','--arm',a,'--seed',str(s)],cwd=OUT,env=e,stdout=l,stderr=subprocess.STDOUT);active[gpu]=(p,a,s,l)
   dump(OUT/'status.json',{'state':'training','stage':stage,'completed':completed,'pending':pending,'active':[{'gpu':g,'pid':p.pid,'arm':a,'seed':s} for g,(p,a,s,l) in active.items()],'updated':time.time()})
   if active:time.sleep(15)
 dump(OUT/'status.json',{'state':'evaluating_all_15_fixed_epoch200','updated':time.time()})
 with (OUT/'summary.log').open('x') as f:code=subprocess.call([sys.executable,'-u',str(OUT/'summarize.py'),'--dataset','pavia'],cwd=OUT,env=env,stdout=f,stderr=subprocess.STDOUT)
 if code:dump(OUT/'status.json',{'state':'evaluation_failed','exit_code':code,'no_automatic_retry':True});return
 d=json.loads((OUT/'pavia/summary.json').read_text());lines=['REPAIRED-NORMALIZED PAVIA: fixed epoch200, 3 predeclared seeds, formal evidence only.','No raw-input results included. No HyRANK or additional variant.']
 for label,name in [('1. Complete method C versus A','C-A'),('2. Center versus local mean reference','C-B'),('3. Local std normalization','C-F'),('4. Capacity-matched nonlinearity','C-G')]:
  z=d['paired_deltas'][name];lines.append(label)
  for k in ['oa_percent','aa_percent','kappa']:lines.append(f"{k}: {z[k]['mean']:+.6f} +/- {z[k]['sample_std']:.6f}; paired seeds={z[k]['values']}; all3positive={z[k]['all_three_positive']}")
  lines.append('Stable positive OA/AA across all seeds: '+str(all(z[k]['all_three_positive'] for k in ['oa_percent','aa_percent'])))
 (OUT/'FINAL_ANSWERS.txt').write_text('\n'.join(lines)+'\n')
 dump(OUT/'status.json',{'state':'completed','formal_normalized_Pavia_runs':15,'seeds':list(SEEDS),'epochs':200,'HyRANK_runs':0,'no_more_experiments':True,'updated':time.time()})
if __name__=='__main__':main()
