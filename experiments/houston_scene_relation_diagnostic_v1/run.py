import json,time,argparse,numpy as np,torch
import common as c

def main(seed):
 c.verify();torch.set_num_threads(2);f=c.frozen();env=f.environment();env[4](seed);model,initial=f.build(seed,'C','houston');config=json.loads((c.BASE/'houston'/('C_%d/config.json'%seed)).read_text());assert f.state_hash(model)==config['initial_full_model_sha256'];path=c.BASE/'houston'/('C_%d/checkpoints/epoch_200.pth'%seed);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);model=model.cuda().eval()
 for p in model.parameters():p.requires_grad_(False)
 before=f.state_hash(model);folder=c.OUT/('seed_%d'%seed);folder.mkdir(exist_ok=False);data=c.AllPixels();loader=torch.utils.data.DataLoader(data,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+127000));start=time.monotonic();logits=np.lib.format.open_memmap(folder/'full_scene_logits.npy',mode='w+',dtype=np.float32,shape=(len(data),7));offset=0
 with torch.inference_mode():
  for batch in loader:
   batch=batch.cuda();z=model(batch,batch)[1].cpu().numpy();assert np.isfinite(z).all();logits[offset:offset+len(z)]=z;offset+=len(z)
   if offset%32768==0:print('DENSE_PROGRESS',seed,offset,'/',len(data),flush=True)
 logits.flush();assert offset==len(data) and f.state_hash(model)==before;inference_seconds=time.monotonic()-start
 z=np.asarray(logits,dtype=np.float64);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);n=np.load(c.OUT/'neighbors.npy');u=np.load(c.OUT/'uniform_weights.npy');w=np.load(c.OUT/'spectral_weights.npy');aggtime=time.monotonic()
 for name,q in [('A',p),('B',c.aggregate(p,n,u)),('C',c.aggregate(p,n,w))]:np.save(folder/('probabilities_'+name+'.npy'),q);np.save(folder/('predictions_'+name+'.npy'),q.argmax(1))
 c.dump(folder/'frozen_manifest.json',{'seed':seed,'checkpoint':str(path),'checkpoint_sha256':c.sha(path),'parameters':sum(p.numel() for p in model.parameters()),'model_state_unchanged':True,'model_state_before_after_sha256':before,'full_scene_pixels':len(data),'inference_seconds':inference_seconds,'aggregation_all3_seconds':time.monotonic()-aggtime,'target_GT_accessed':False,'files':{str(p):c.sha(p) for p in folder.glob('*.npy')}});c.verify();print('FULL_SCENE_FROZEN',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=c.SEEDS,required=True);main(p.parse_args().seed)
