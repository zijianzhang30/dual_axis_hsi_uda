import argparse,json,time
import numpy as np,torch
import study as s
from anchor import install

def main(arm):
 s.verify();torch.set_num_threads(2);c=s.environment();m,ls,helper,n,audit=c.setup('houston','C',s.SEED);install(m,arm);ds=ls[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);m.cuda().eval();folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==100;pred=folder/'offline_predictions';pred.mkdir(exist_ok=False);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(187100));records=[];start=time.monotonic()
 for r in mf['checkpoints']:
  assert s.sha(r['path'])==r['sha256'];m.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['model']);before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert c.state_hash(m)==before;path=pred/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(path,logits=z,predictions=p,coordinates=ds.indices);records.append({'epoch':r['epoch'],'path':str(path),'sha256':s.sha(path),'target_label_values_zero':True});print('PREDICTION_FROZEN',arm,r['epoch'],flush=True)
 assert [r['epoch'] for r in records]==s.PLAN['grid'];s.dump(folder/'offline_predictions_frozen.json',{'records':records,'target_GT_not_scored':True,'model_and_running_stats_unchanged':True,'seconds':time.monotonic()-start});s.verify()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);main(p.parse_args().arm)
