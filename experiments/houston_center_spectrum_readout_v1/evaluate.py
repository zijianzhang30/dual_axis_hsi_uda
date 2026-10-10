import argparse,json,time
import numpy as np,torch
import study as s
from readout import install

def main(arm,seed):
 s.verify();torch.set_num_threads(2);c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',seed);install(m,arm,seed);ds=ls[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);m.cuda().eval();folder=s.OUT/('%s_%d'%(arm,seed));mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==200;pred=folder/'offline_predictions';pred.mkdir(exist_ok=False);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+185000));records=[];start=time.monotonic()
 for r in mf['checkpoints']:
  assert s.sha(r['path'])==r['sha256'];m.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['model']);before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert c.state_hash(m)==before;path=pred/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(path,logits=z,predictions=p);records.append({'epoch':r['epoch'],'path':str(path),'sha256':s.sha(path),'target_label_values_zero':True});print('READOUT_PREDICTION_FROZEN',arm,seed,r['epoch'],flush=True)
 assert [r['epoch'] for r in records]==s.PLAN['evaluation_grid'];s.dump(folder/'offline_predictions_frozen.json',{'records':records,'target_GT_not_scored':True,'running_statistics_unchanged':True,'seconds':time.monotonic()-start});s.verify();print('READOUT_EVALUATION_COMPLETE',arm,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);a=p.parse_args();main(a.arm,a.seed)
