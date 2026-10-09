import json,numpy as np
import study as s

def main():
 c,ns=s.environment();args=c.options(2100);ds=ns['loaders'](args)[0][-1].dataset
 labels={'native':np.asarray(ds.labels,dtype=np.int64),'full':s.geometry().full_geometry(ds)[-1]}
 paths={'R0':s.R0/'C_2100/frozen_manifest.json',**{a:s.OUT/('%s_2100/frozen_manifest.json'%a) for a in s.ARMS}}
 results=[]
 for arm,path in paths.items():
  if arm=='R0' and not path.exists():
   row={scope+'_prediction_path':str(s.R0/'C_2100/predictions'/('%s_epoch_100.npz'%scope)) for scope in ('native','full')}
  else:
   m=json.loads(path.read_text());row=next(r for r in m['evaluations'] if r['epoch']==100)
  for scope in ('native','full'):
   z=np.load(row[scope+'_prediction_path']);metrics=c.helper.metrics(labels[scope],z['predictions'],z['logits'],12)
   results.append({'arm':arm,'scope':scope,'epoch':100,'metrics':metrics,
   'zero_recall_classes':[k+1 for k,v in enumerate(metrics['per_class_percent']) if v==0],
   'single_class_collapse_at95pct':max(metrics['prediction_shares'])>=.95})
 deltas=[]
 for scope in ('native','full'):
  for baseline in ('R0','R1'):
   a=next(r['metrics'] for r in results if r['arm']=='R2' and r['scope']==scope);b=next(r['metrics'] for r in results if r['arm']==baseline and r['scope']==scope)
   deltas.append({'scope':scope,'comparison':'R2 minus '+baseline,**{k:a[k]-b[k] for k in ('oa_percent','aa_percent','kappa')}})
 diagnostics={}
 for arm in s.ARMS:
  folder=s.OUT/('%s_2100'%arm);rows=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
  grads=[r['adapter_gradient_norms_last_step'] for r in rows]
  diagnostics[arm]={'last_step_gradient_nonzero_epochs':{k:sum(g[k]>0 for g in grads) for k in grads[0]},
  'parameters':json.loads((folder/'config.json').read_text())['parameters'],
  'gradient_limitation':'epoch last-step measurements, not every-step history',
  'train_seconds':json.loads((folder/'frozen_manifest.json').read_text())['train_seconds']}
 s.dump(s.OUT/'summary.json',{'results':results,'deltas':deltas,'adapter_diagnostics':diagnostics,
 'R0_reused_no_retraining':True,'total_extra_parameters_each':12288,'per_domain_width_R1_32_R2_16':True,
 'fixed100_no_target_best':True,'scope':'exploratory only, no automatic3seed or method claim'})
 print(json.dumps({'results':results,'deltas':deltas},indent=2),flush=True)
if __name__=='__main__':main()
