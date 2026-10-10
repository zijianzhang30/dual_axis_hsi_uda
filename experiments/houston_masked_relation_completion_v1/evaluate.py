import argparse,json,time
import numpy as np,torch
import study as s

def main(arm):
 s.verify();torch.set_num_threads(2);c=s.env();m,ls,helper,classes,audit=c.setup('houston','C',2100);ds=ls[-1].dataset;ds.labels=np.zeros_like(ds.labels);ds.label=np.zeros_like(ds.label);m.cuda().eval();folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==100;out=folder/'offline_predictions';out.mkdir(exist_ok=False);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(199100));rs=[];start=time.monotonic()
 for r in mf['checkpoints']:
  assert s.sha(r['path'])==r['sha256'];m.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['model']);before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert c.state_hash(m)==before;f=out/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(f,logits=z,predictions=p);rs.append({'epoch':r['epoch'],'path':str(f),'sha256':s.sha(f)});print('COMPLETION_PREDICTION_FROZEN',arm,r['epoch'],flush=True)
 s.dump(folder/'offline_predictions_frozen.json',{'records':rs,'decoder_used':False,'target_GT_scored':False,'BN_buffers_unchanged':True,'seconds':time.monotonic()-start});s.verify()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);main(p.parse_args().arm)
