"""Frozen inference only: no training, no target GT in predictor, all groups same20 checkpoints."""
import argparse,json
import numpy as np
import torch
import study as s
import adapter as a

def main(arm,seed):
 torch.set_num_threads(2);c=s.environment();model,loaders,helper,classes,audit=c.setup('hyrank','C',seed)
 if arm!='R0':a.install(model,arm,seed)
 model=model.cuda().eval();ds=loaders[-1].dataset;clean=c.clean_dataset(ds)
 ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
 def loader(d):return torch.utils.data.DataLoader(d,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+85000))
 native_loader=loader(ds);full_loader=loader(clean)
 if arm=='R0':folder=s.OUT/('R0_%d'%seed);folder.mkdir(exist_ok=False);manifest=json.loads((s.HIST/'hyrank'/('C_LMMD_%d/frozen_manifest.json'%seed)).read_text())
 else:folder=s.OUT/('%s_%d'%(arm,seed));manifest=json.loads((folder/'frozen_manifest.json').read_text())
 pred_dir=folder/'offline_predictions';pred_dir.mkdir(exist_ok=False);records=[]
 for r in manifest['checkpoints']:
  if r['epoch']%10:continue
  saved=torch.load(r['path'],map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);before=helper.state_hash(model);record={'epoch':r['epoch']}
  with a.eval_domain(model,'target'):
   for scope,dl in [('native',native_loader),('full',full_loader)]:
    logits,pred=helper.predict(model,dl,torch.device('cuda:0'));p=pred_dir/('%s_epoch_%03d.npz'%(scope,r['epoch']));np.savez_compressed(p,logits=logits,predictions=pred);record[scope+'_prediction_path']=str(p);record[scope+'_sha256']=s.sha(p)
    if arm=='R0' and scope=='native' and r['epoch']==200:
     old=np.load(manifest['prediction']['path']);assert np.array_equal(pred,old['predictions']);record['historical_fixed200_predictions_exact']=True
  assert before==helper.state_hash(model);records.append(record)
 assert len(records)==20
 np.save(folder/'native_indices.npy',ds.indices);np.save(folder/'full_indices.npy',clean.indices)
 s.dump(folder/'offline_predictions_frozen.json',{'arm':arm,'seed':seed,'evaluations':records,'target_GT_values_zero_for_all_predictions':True,
 'grid':'10:10:200','no_training_R0':arm=='R0','R0_historical_reuse':str(s.HIST/'hyrank'/('C_LMMD_%d'%seed)) if arm=='R0' else None})
 print('OFFLINE_PREDICTIONS_COMPLETE',arm,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('R0',)+s.ARMS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);args=p.parse_args();main(args.arm,args.seed)
