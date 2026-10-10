import argparse,json,numpy as np,torch,time
import common as c

def main(dataset,seed):
 c.verify();torch.set_num_threads(2);env=c.environment(dataset);job=c.OUT/('%s_%d'%(dataset,seed));job.mkdir(exist_ok=False);epochs=(20,200) if dataset=='houston' else (10,100)
 if dataset=='houston':
  olddata=c.B/'tokenizer_stem_value_diagnostic_v1/data';mask=np.load(olddata/'source_probe_train_mask.npy');sc=np.load(olddata/'source_coords.npy')[mask];sy=np.load(olddata/'source_labels.npy')[mask];tc=np.load(olddata/'target_coords.npy');sets={'source':c.HoustonPatches('source',sc),'target':c.HoustonPatches('target',tc)};bs=128
 else:
  b=env.base()
  with np.load(env.P2/'inputs'/('source_selection_%d.npz'%seed)) as z:sc=z['coords'];sy=z['labels']
  tc=np.load(env.P2/'inputs/target_training_coords.npy');sets={'source':b.PatchSet('P2','source',sc),'target':b.PatchSet('P2','target',tc)};bs=32
 for name,a in [('source_coords',sc),('source_labels',sy),('target_coords',tc)]:np.save(job/(name+'.npy'),a)
 for objective in ('CE','LMMD'):
  for epoch in epochs:
   cell=job/('%s_%03d'%(objective,epoch));cell.mkdir();paths={};start=time.monotonic();path=c.folder(dataset,objective,seed)/'checkpoints'/('epoch_%03d.pth'%epoch)
   if dataset=='houston' and objective=='CE' and epoch==200:
    old=c.B/'tokenizer_stem_value_diagnostic_v1'/('seed_%d'%seed)
    for domain in ('source','target'):
     paths[domain]={}
     for stage in c.STAGES+('logits',):
      p=old/(domain+'_'+stage+'.npy')
      if domain=='source':q=cell/(domain+'_'+stage+'.npy');np.save(q,np.load(p,mmap_mode='r')[mask]);p=q
      paths[domain][stage]=str(p)
    reused=True;immutable=True
   else:
    if dataset=='houston':
     f,worker,helper=env.environment('houston');worker(seed);model,_=f.build(seed,'C','houston');statehash=f.state_hash;backbone=model
    else:model,_=env.build(seed,'C');statehash=env.state_hash;backbone=model.backbone
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model']);model=model.cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    before=statehash(model);latest={};hooks=[]
    def hook(key):
     def cb(mod,inp,out):latest[key]=out.detach()
     return cb
    hooks.append(backbone.conv3d_features[2].register_forward_hook(hook('F3D_map')));hooks.append(backbone.conv2d_features[2].register_forward_hook(hook('V_map')));hooks.append(backbone.norm.register_forward_hook(hook('Z_map')));original=backbone._forward_semantic_tokens
    def capture(value):
     t=original(value);latest['T']=t.detach();return t
    backbone._forward_semantic_tokens=capture
    with torch.inference_mode():
     for domain,ds in sets.items():
      arrays={};paths[domain]={};offset=0;loader=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+129000))
      for x,dummy in loader:
       n=len(x)
       if dataset=='pavia' and n<32:x=torch.cat((x,x[-1:].expand(32-n,-1,-1,-1,-1)))
       x=x.cuda();r=model(x,x);z=r['lt'] if dataset=='pavia' else r[1];v=latest['V_map'];f3=latest['F3D_map'];cy,cx=v.shape[-2]//2,v.shape[-1]//2;values={'F3D':f3[...,cy,cx].flatten(1),'V':v[...,cy,cx],'V_mean':v.mean((-2,-1)),'T':latest['T'].flatten(1),'Z':latest['Z_map'][:,0],'logits':z}
       for stage,value in values.items():
        a=value[:n].cpu().numpy();assert np.isfinite(a).all()
        if stage not in arrays:
         p=cell/(domain+'_'+stage+'.npy');arrays[stage]=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(len(ds),a.shape[1]));paths[domain][stage]=str(p)
        arrays[stage][offset:offset+n]=a
       offset+=n
      assert offset==len(ds)
      for arr in arrays.values():arr.flush()
    backbone._forward_semantic_tokens=original
    for h in hooks:h.remove()
    immutable=before==statehash(model);assert immutable;del model,backbone,arrays,latest;torch.cuda.empty_cache();reused=False
   c.dump(cell/'frozen_manifest.json',{'dataset':dataset,'seed':seed,'objective':objective,'epoch':epoch,'features':paths,'checkpoint':str(path),'checkpoint_sha256':c.sha(path),'state_unchanged':immutable,'reused_existing_features':reused,'target_GT_accessed':False,'seconds':time.monotonic()-start,'files':{p:c.sha(p) for domain in paths.values() for p in domain.values()}});print('CELL_FROZEN',dataset,seed,objective,epoch,'reused',reused,flush=True)
 c.verify();c.dump(job/'complete.json',{'completed':True,'no_training':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('houston','pavia'),required=True);p.add_argument('--seed',type=int,choices=c.SEEDS,required=True);a=p.parse_args();main(a.dataset,a.seed)
