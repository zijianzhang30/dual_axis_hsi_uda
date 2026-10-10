import argparse,json,time
import numpy as np,torch
import study as s
from attention import install

def main(arm):
 s.verify();torch.set_num_threads(2);c=s.env();m,ls,helper,classes,ba=c.setup('houston','C',2100);install(m,arm);m.cuda().eval();att=m.blocks[0].attn;folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==200;ds=ls[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(198000));pred=folder/'offline_predictions';pred.mkdir(exist_ok=False);records=[];start=time.monotonic()
 for r in mf['checkpoints']:
  assert s.sha(r['path'])==r['sha256'];state=torch.load(r['path'],map_location='cpu',weights_only=True);assert state['config']['arm']==arm;m.load_state_dict(state['model']);att.talking_log=False;before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert c.state_hash(m)==before and np.isfinite(z).all()
  with torch.no_grad():
   x=torch.stack([ds[i][0] for i in range(128)]).cuda();att.talking_log=True;m(x,x);diag=dict(att.last_diagnostics);att.talking_log=False
  assert c.state_hash(m)==before;path=pred/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(path,logits=z,predictions=p);records.append({'epoch':r['epoch'],'path':str(path),'sha256':s.sha(path),'first128_eval_attention_diagnostics':diag,'target_label_values_zero':True});print('TALKING_PREDICTION_FROZEN',arm,r['epoch'],flush=True)
 s.dump(folder/'offline_predictions_frozen.json',{'records':records,'target_GT_scored':False,'BN_buffers_unchanged':True,'seconds':time.monotonic()-start});s.verify();print('TALKING_EVALUATION_COMPLETE',arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);main(p.parse_args().arm)
