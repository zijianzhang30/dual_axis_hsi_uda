import argparse,csv,hashlib,json,time
from argparse import Namespace
import numpy as np
import torch
import torch.nn.functional as F
import study as s
from boundary import loss,perturbed

def main(arm,seed):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed']
 torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda:0')
 c=s.environment();env=c.environment();env[4](seed)
 args=Namespace(dataset='houston',seed=seed,num_workers=4,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
 source,val,target,target_eval=c.loaders(args);model,initial=c.build(seed,'C','houston')
 old=json.loads((s.BASE/'houston'/f'C_{seed}'/'config.json').read_text())
 idx={name:hashlib.sha256(l.dataset.indices.tobytes()).hexdigest() for name,l in [('source',source),('val',val),('target',target),('target_eval',target_eval)]}
 assert initial==old['initial_original_model_sha256'] and c.state_hash(model)==old['initial_full_model_sha256'] and idx==old['indices_sha256']
 assert sum(p.numel() for p in model.parameters())==old['parameters']
 for loader in [target,target_eval]:
  loader.dataset.labels=np.zeros_like(loader.dataset.labels);loader.dataset.label=np.zeros_like(loader.dataset.label)
 model=model.to(device);opt=torch.optim.SGD(model.parameters(),lr=.01)
 weights=None if arm=='B0' else torch.as_tensor(np.load(s.OUT/'weights'/f'{arm}_{seed}.npz')['matrix'],device=device,dtype=torch.float32)
 folder=s.OUT/f'{arm}_{seed}';folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir()
 config={'arm':arm,'seed':seed,'baseline_config':old,'initial_full_model_sha256':c.state_hash(model),'indices_sha256':idx,'epochs':200,
  'loss':'source CE' if arm=='B0' else 'source CE + 1*weighted pairwise source perturbed CE','fraction':.5,'boundary_loss_weight':1.,
  'weights_sha256':None if arm=='B0' else s.sha(s.OUT/'weights'/f'{arm}_{seed}.npz'),'pretrain_manifest_sha256':s.sha(s.OUT/'PRETRAIN_WEIGHTS_CODE_FROZEN.json'),
  'target_label_values_zeroed_after_native_loader':True,'target_GT_scored':False,'parameters':old['parameters']}
 s.dump(folder/'config.json',config)
 ref=[json.loads(line) for line in (s.BASE/'houston'/f'C_{seed}'/'history.jsonl').read_text().splitlines()]
 captures=[]
 def capture(mod,inputs):captures.append(inputs[0])
 hook=model.nn1.register_forward_pre_hook(capture);checkpoints=[];start=time.monotonic()
 print('START',arm,seed,flush=True)
 with (folder/'history.jsonl').open('x') as history:
  for epoch in range(1,201):
   model.train();bhash=hashlib.sha256();rhash=hashlib.sha256();steps=count=0;ce_sum=rob_sum=0.;margin_sum=pert_margin_sum=0.;half_error=0.
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for tensor in (x,y,t):bhash.update(tensor.numpy().tobytes())
    rhash.update(torch.get_rng_state().numpy().tobytes());rhash.update(torch.cuda.get_rng_state().numpy().tobytes())
    x,y,t=x.to(device),y.to(device),t.to(device);opt.zero_grad(set_to_none=True);captures.clear();out=model(x,t)
    assert len(captures)==4;z=captures[0];assert z.shape==(len(y),64)
    ce=F.cross_entropy(out[0],y)
    if arm=='B0':rob=ce.new_zeros(())
    else:
     rob,pl,m,n=loss(z,y,model.nn1,weights)
     with torch.no_grad():
      mask=~F.one_hot(y,7).bool();pm=pl.gather(2,y[:,None,None].expand(-1,7,1)).squeeze(-1)-pl.diagonal(dim1=1,dim2=2)
      e=float((pm[mask]-.5*m[mask]).abs().max());assert e<5e-5
      half_error=max(half_error,e);margin_sum+=float((m/n)[mask].sum());pert_margin_sum+=float((pm/n)[mask].sum())
    objective=ce+rob;assert torch.isfinite(objective);objective.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step();steps+=1;count+=len(y);ce_sum+=float(ce.detach())*len(y);rob_sum+=float(rob.detach())*len(y);captures.clear()
   assert steps==18 and bhash.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rhash.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('stream mismatch',arm,seed,epoch)
   bn={k:int(v.num_batches_tracked) for k,v in model.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==epoch*18 for v in bn.values())
   row={'epoch':epoch,'steps':steps,'source_ce':ce_sum/count,'boundary_ce':rob_sum/count,'total_loss':(ce_sum+rob_sum)/count,
    'source_signed_pair_geometric_margin':margin_sum/(count*6) if arm!='B0' else None,'perturbed_signed_pair_geometric_margin':pert_margin_sum/(count*6) if arm!='B0' else None,
    'half_margin_max_error':half_error,'batch_stream_sha256':bhash.hexdigest(),'original_rng_stream_sha256':rhash.hexdigest(),'BN_counts':bn,'elapsed_seconds':time.monotonic()-start}
   if epoch==1 or epoch%10==0:
    captures.clear();row['source_val']=env[6].source_evaluate(model,val,device);captures.clear()
    print(json.dumps({'arm':arm,'seed':seed,**row}),flush=True)
   if epoch%10==0:
    path=folder/'checkpoints'/f'epoch_{epoch:03d}.pth';torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':s.sha(path)})
   history.write(json.dumps(row)+'\n');history.flush()
 train_seconds=time.monotonic()-start
 # Original fixed200 inference, no target score. No training resumes after this.
 model.eval();before=c.state_hash(model);all_logits=[]
 with torch.inference_mode():
  for images,dummy in target_eval:
   assert not dummy.any();captures.clear();images=images.to(device);logits=model(images,images)[1];all_logits.append(logits.cpu().numpy())
 captures.clear();logits=np.concatenate(all_logits);np.savez_compressed(folder/'predictions_200.npz',logits=logits,predictions=logits.argmax(1))
 assert before==c.state_hash(model)
 np.savez_compressed(folder/'frozen_head.npz',weight=model.nn1.weight.detach().cpu().numpy(),bias=model.nn1.bias.detach().cpu().numpy())
 source_diagnostics(model,captures,device,folder,source.dataset.data)
 assert before==c.state_hash(model);hook.remove()
 replay=None
 if arm=='B0':
  archived=torch.load(s.BASE/'houston'/f'C_{seed}'/'checkpoints/epoch_200.pth',map_location='cpu',weights_only=True)['model']
  ap=np.load(s.BASE/'houston'/f'C_{seed}'/'predictions_200.npz')
  replay={'state_exact':all(torch.equal(v.cpu(),archived[k]) for k,v in model.state_dict().items()),
   'max_state_error':max(float((v.cpu()-archived[k]).abs().max()) for k,v in model.state_dict().items()),
   'predictions_exact':bool(np.array_equal(logits.argmax(1),ap['predictions'])),'max_logit_error':float(np.abs(logits-ap['logits']).max())}
 manifest={'arm':arm,'seed':seed,'completed_epochs':200,'steps':3600,'train_seconds':train_seconds,'checkpoint_records':checkpoints,'source_val_final':row['source_val'],
  'archived_B0_replay':replay,'target_GT_scored':False,'files_sha256':{str(p.relative_to(folder)):s.sha(p) for p in folder.rglob('*') if p.is_file()}}
 s.dump(folder/'FINAL_PREDICTIONS_FROZEN.json',manifest);print('COMPLETE',arm,seed,'seconds',train_seconds,'replay',replay,flush=True)

def source_diagnostics(model,captures,device,folder,cube):
 coords=np.load(s.DATA/'source_coords.npy');y=np.load(s.DATA/'source_labels.npy');train=np.load(s.DATA/'source_probe_train_mask.npy');zs=[];ls=[]
 with torch.inference_mode():
  for start in range(0,len(coords),128):
   batch=coords[start:start+128];patches=np.stack([cube[r-6:r+7,c-6:c+7].transpose(2,0,1)[None] for r,c in batch]).astype(np.float32)
   x=torch.as_tensor(patches,device=device);captures.clear();out=model(x,x)[1];assert len(captures)==1
   zs.append(captures[0].cpu().numpy());ls.append(out.cpu().numpy())
  z=np.concatenate(zs);logits=np.concatenate(ls);captures.clear();np.savez_compressed(folder/'source_frozen.npz',rawZ=z,logits=logits,predictions=logits.argmax(1),labels=y,coords=coords,train_mask=train)
 source_margin_export(model,device,folder,z,logits,y,train)

def source_margin_export(model,device,folder,z,logits,y,train):
 with torch.inference_mode():
  zz=torch.as_tensor(z,device=device);yy=torch.as_tensor(y,device=device,dtype=torch.long);zp,margin,norm=perturbed(zz,yy,model.nn1);pl=F.linear(zp,model.nn1.weight,model.nn1.bias)
  pm=pl.gather(2,yy[:,None,None].expand(-1,7,1)).squeeze(-1)-pl.diagonal(dim1=1,dim2=2)
  mask=~F.one_hot(yy,7).bool();assert float((pm[mask]-.5*margin[mask]).abs().max())<5e-5
  other=logits.copy();other[np.arange(len(y)),y]=-np.inf;original_m=logits[np.arange(len(y)),y]-other.max(1)
  pln=pl.cpu().numpy();own=pln[np.arange(len(y))[:,None],np.arange(7)[None,:],y[:,None]];otherp=pln.copy();otherp[np.arange(len(y))[:,None],np.arange(7)[None,:],y[:,None]]=-np.inf;pert_m=own-otherp.max(2)
  geom=(margin/norm).cpu().numpy();pgeom=(pm/norm).cpu().numpy();pertpred=pln.argmax(2);rawpred=logits.argmax(1)
  np.savez_compressed(folder/'source_margin_arrays.npz',signed_pair_geometric_margin=geom,perturbed_signed_pair_geometric_margin=pgeom,original_multiclass_logit_margin=original_m,perturbed_multiclass_logit_margin=pert_m,perturbed_predictions=pertpred)
 rows=[]
 for split,sm in [('all',np.ones(len(y),bool)),('train',train),('val',~train)]:
  for cls in [None,*range(7)]:
   idx=sm if cls is None else sm&(y==cls);pairmask=(y[:,None]!=np.arange(7)[None,:])&idx[:,None]
   rows.append({'split':split,'class':'macro_samples' if cls is None else f'C{cls+1}','n':int(idx.sum()),'original_OA':100*float((rawpred[idx]==y[idx]).mean()),
    'perturbed_OA':100*float((pertpred==y[:,None])[pairmask].mean()),'original_true_vs_strongest_logit_margin_mean':float(original_m[idx].mean()),
    'perturbed_true_vs_strongest_logit_margin_mean':float(pert_m[pairmask].mean()),'original_signed_pair_geometric_margin_mean':float(geom[pairmask].mean()),'perturbed_signed_pair_geometric_margin_mean':float(pgeom[pairmask].mean())})
 with (folder/'source_margin_summary.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 pairrows=[]
 for a,b in s.PAIRS:
  for direction in [(a,b),(b,a)]:
   c,d=direction;idx=y==c
   pairrows.append({'pair':f'C{a+1}/C{b+1}','true_class':f'C{c+1}','competitor':f'C{d+1}','n':int(idx.sum()),
    'original_geometric_margin_mean':float(geom[idx,d].mean()),'perturbed_geometric_margin_mean':float(pgeom[idx,d].mean()),
    'original_multiclass_margin_mean':float(original_m[idx].mean()),'perturbed_multiclass_margin_mean':float(pert_m[idx,d].mean()),
    'original_OA':100*float((rawpred[idx]==c).mean()),'perturbed_OA':100*float((pertpred[idx,d]==c).mean())})
 with (folder/'source_pair_margin_summary.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pairrows[0]));w.writeheader();w.writerows(pairrows)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);args=p.parse_args();main(args.arm,args.seed)
