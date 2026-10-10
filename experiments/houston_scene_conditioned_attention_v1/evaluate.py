import argparse,json,time
import numpy as np,torch
import study as s
from mechanism import install

def main(arm):
 s.verify();torch.set_num_threads(2);c=s.env();m,ls,helper,classes,ba=c.setup('houston','C',2100);install(m,arm);m.cuda().eval();att=m.blocks[0].attn;folder=s.OUT/(arm+'_2100');mf=json.loads((folder/'frozen_manifest.json').read_text());assert mf['completed_epochs']==100;ds=ls[-1].dataset;ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels);loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(185000));pred=folder/'offline_predictions';pred.mkdir(exist_ok=False);records=[];start=time.monotonic()
 for r in mf['checkpoints']:
  assert s.sha(r['path'])==r['sha256'];state=torch.load(r['path'],map_location='cpu',weights_only=True);m.load_state_dict(state['model']);att.scene_summary_mode='fixed';att.scene_log=False;before=c.state_hash(m);z,p=helper.predict(m,loader,torch.device('cuda:0'));assert c.state_hash(m)==before
  with torch.no_grad():
   x=torch.stack([ds[i][0] for i in range(128)]).cuda();att.scene_log=True;fixed=m(x,x)[1];diag=dict(att.last_diagnostics);saved=att.scene_summary.clone();att.scene_summary_mode='batch';batch=m(x,x)[1];context=att.last_context.detach();diag.update({'fixed_vs_current_eval_targetbatch_logits_max_difference':float((fixed-batch).abs().max()),'fixed_vs_batch_prediction_disagreement_count':int((fixed.argmax(1)!=batch.argmax(1)).sum()),'fixed_vs_targetbatch_summary_L2':float((saved-context).norm()),'comparison_pixels':128,'S1_batch_alternative_is_target_context_not_source':arm=='S1'});att.scene_summary_mode='fixed';att.scene_log=False
  assert c.state_hash(m)==before;path=pred/('epoch_%03d.npz'%r['epoch']);np.savez_compressed(path,logits=z,predictions=p);records.append({'epoch':r['epoch'],'path':str(path),'sha256':s.sha(path),'diagnostics':diag,'target_label_values_zero':True});print('SCENE_PREDICTION_FROZEN',arm,r['epoch'],flush=True)
 s.dump(folder/'offline_predictions_frozen.json',{'records':records,'target_GT_scored':False,'saved_fixed_scene_summary_used':True,'BN_buffers_unchanged':True,'seconds':time.monotonic()-start});s.verify();print('SCENE_EVALUATION_COMPLETE',arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);main(p.parse_args().arm)
