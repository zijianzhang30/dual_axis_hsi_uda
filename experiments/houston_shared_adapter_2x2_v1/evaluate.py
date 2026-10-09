import argparse,json,numpy as np,torch
import study as s
import adapter as a

def evaluate(group,seed):
 torch.set_num_threads(2);c=s.environment();m,loaders,helper,classes,audit=c.setup('houston','C',seed)
 added=group.endswith('1')
 if added:a.install(m,'R1',seed)
 m=m.cuda().eval();folder=s.OUT/('%s_%d'%(group,seed));origin=folder if added else s.baseline(group,seed)
 manifest=json.loads((origin/'frozen_manifest.json').read_text());assert len(manifest['checkpoints'])>=20
 if not added:folder.mkdir(exist_ok=False)
 pred=folder/'offline_predictions';pred.mkdir(exist_ok=False);ds=loaders[-1].dataset;idx=ds.indices.copy();ds.labels=np.zeros_like(ds.labels);ds.label=np.zeros_like(ds.label)
 loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+85000))
 np.save(folder/'evaluation_indices.npy',idx);records=[]
 for item in manifest['checkpoints']:
  epoch=item['epoch']
  if epoch%10:continue
  assert s.sha(item['path'])==item['sha256'];m.load_state_dict(torch.load(item['path'],map_location='cpu',weights_only=True)['model']);before=c.state_hash(m)
  with a.eval_domain(m,'target'):logits,predictions=helper.predict(m,loader,torch.device('cuda:0'))
  assert before==c.state_hash(m)
  if not added and epoch==200:assert np.array_equal(predictions,np.load(manifest['prediction']['path'])['predictions']),('baseline replay',group,seed)
  path=pred/('epoch_%03d.npz'%epoch);np.savez_compressed(path,logits=logits,predictions=predictions)
  records.append({'epoch':epoch,'path':str(path),'sha256':s.sha(path)})
 s.dump(folder/'offline_predictions_frozen.json',{'group':group,'seed':seed,'records':records,'no_target_GT_scoring':True});print('OFFLINE_COMPLETE',group,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',choices=s.GROUPS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);args=p.parse_args();evaluate(args.group,args.seed)
