import argparse,hashlib,json,pickle,random,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
from expert import build,DropoutReplay

def rng(generator):
 return {'shared_loader':hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'torch_cuda':hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),'python':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()}
@torch.inference_mode()
def source_val(e,loader):
 e.eval();n=hit=0;loss=0.
 for batch in loader:
  x,y=batch[:2];z=e(x[:,0,:,6,6].cuda());y=y.cuda();n+=len(y);hit+=int((z.argmax(1)==y).sum());loss+=float(F.cross_entropy(z,y))*len(y)
 return {'OA':hit/n*100,'CE':loss/n}
def main(dataset,seed):
 s.verify();pf=json.loads((s.OUT/('preflight_'+dataset+'.json')).read_text());assert pf['passed'];torch.set_num_threads(2);torch.cuda.set_device(0);m,values,r=s.setup(dataset,seed);source,val,target,ev=values;e=build(r['bands'],r['classes'],seed).cuda();opt=torch.optim.SGD(e.parameters(),lr=.01);assert not {id(p) for p in m.parameters()} & {id(p) for g in opt.param_groups for p in g['params']}
 # The original main-network initialization/RNG draw order has already occurred.
 m.cuda();m.load_state_dict(torch.load(s.baseline_records(dataset,seed)[-1]['checkpoint'],map_location='cpu',weights_only=True)['model']);m.eval().requires_grad_(False);frozen=r['hash'](m);assert all(int(v.num_batches_tracked)==200*r['steps'] for v in m.modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)))
 if dataset=='hyrank':
  radius=6;xx,yy=np.nonzero(ev.dataset.label>=0);keep=(xx>=radius)&(xx<ev.dataset.data.shape[0]-radius)&(yy>=radius)&(yy<ev.dataset.data.shape[1]-radius);fullcoords=np.column_stack((xx[keep],yy[keep]));assert len(fullcoords)==10317
 else:fullcoords=ev.dataset.indices.copy()
 for ds in (target.dataset,ev.dataset):ds.label=np.zeros_like(ds.label);ds.labels=np.zeros_like(ds.labels)
 replay=DropoutReplay(pf['dropout_calls']);f=s.OUT/('%s_%d'%(dataset,seed));f.mkdir(exist_ok=False);(f/'checkpoints').mkdir();np.savez_compressed(f/'coordinates_frozen.npz',native=ev.dataset.indices,full=fullcoords);s.dump(f/'config.json',{'dataset':dataset,'seed':seed,'plan':s.PLAN,'runtime':{k:v for k,v in r.items() if k in ('bands','classes','steps','initial','indices','data_protocol')},'spectral_parameters':sum(p.numel() for p in e.parameters()),'initial_spectral_hash':r['hash'](e),'relation_checkpoint_reused':s.baseline_records(dataset,seed)[-1],'relation_frozen_hash':frozen,'relation_BN_training_updates_in_original':200*r['steps'],'new_relation_BN_updates':0,'optimizer_excludes_R0':True})
 reference=[json.loads(x) for x in (s.baseline(dataset,seed)/'history.jsonl').read_text().splitlines()];checkpoints=[];start=time.monotonic()
 batch_stream=(s.baseline(dataset,seed)/'actual_batches.jsonl').open() if dataset=='hyrank' else None
 with (f/'history.jsonl').open('x') as out:
  for epoch in range(1,201):
   e.train();bh=hashlib.sha256();rh=hashlib.sha256();ah=hashlib.sha256();ih=hashlib.sha256();steps=n=hit=0;total=0.;gradnorm=None
   for sb,tb in zip(source,target):
    x,y=sb[:2];t,dummy=tb[:2]
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bh.update(v.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes())
    if dataset=='hyrank':
     current=rng(source.generator);ah.update(json.dumps(current,sort_keys=True).encode());si,ti=sb[2],tb[2];src=source.dataset.indices[si.numpy()];tar=target.dataset.indices[ti.numpy()];ih.update(src.tobytes());ih.update(tar.tobytes());expected=json.loads(next(batch_stream));assert expected['epoch']==epoch and expected['step_in_epoch']==steps+1;assert expected['source_dataset_indices']==si.tolist() and expected['target_dataset_indices']==ti.tolist();assert expected['rng']==current,('all_rng_mismatch',dataset,seed,epoch,steps,current,expected['rng'])
    # Replay only original model's stochastic draws, independently of spectrum.
    replay.advance();cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();opt.zero_grad(set_to_none=True);z=e(x[:,0,:,6,6].detach().cuda());y=y.cuda();loss=F.cross_entropy(z,y);assert torch.isfinite(loss);loss.backward();opt.step();assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());assert all(p.grad is None for p in m.parameters())
    if steps==0:gradnorm=float(e[1].weight.grad.norm())
    steps+=1;n+=len(y);hit+=int((z.argmax(1)==y).sum());total+=float(loss.detach())*len(y)
   ref=reference[epoch-1];assert steps==r['steps'] and bh.hexdigest()==ref['batch_stream_sha256'];key='original_rng_stream_sha256' if dataset=='pavia' else 'rng_stream_sha256';assert rh.hexdigest()==ref[key],('epoch_rng_mismatch',dataset,seed,epoch)
   if dataset=='hyrank':assert ih.hexdigest()==ref['actual_centers_sha256'] and ah.hexdigest()==ref['all_rng_stream_sha256']
   assert r['hash'](m)==frozen;row={'dataset':dataset,'seed':seed,'epoch':epoch,'steps':steps,'spectral_CE':total/n,'spectral_source_OA':hit/n*100,'first_batch_gradient_norm':gradnorm,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'actual_centers_sha256':ih.hexdigest() if dataset=='hyrank' else None,'all_rng_stream_sha256':ah.hexdigest() if dataset=='hyrank' else None,'relation_hash_unchanged':True,'relation_gradients_None':True,'new_relation_BN_updates':0,'elapsed_seconds':time.monotonic()-start}
   if dataset=='pavia' and (epoch==1 or epoch%10==0):row['spectral_source_val']=source_val(e,val)
   if dataset=='hyrank' and epoch%10==0:
    # Preserve native evaluation-loader iteration/shared generator consumption;
    # no pseudo-label/cache is constructed and no target value is a training input.
    for eb in ev:assert not eb[1].any()
   if epoch%10==0:
    cp=f/'checkpoints'/('spectral_epoch_%03d.pth'%epoch);torch.save({'model':e.state_dict(),'epoch':epoch,'config':json.loads((f/'config.json').read_text())},cp);checkpoints.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp)})
   out.write(json.dumps(row)+'\n');out.flush()
   if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
 if batch_stream is not None:assert next(batch_stream,None) is None;batch_stream.close()
 s.verify();s.dump(f/'SPECTRAL_TRAINING_COMPLETE.json',{'dataset':dataset,'seed':seed,'epochs':200,'updates':200*r['steps'],'checkpoints':checkpoints,'all200_augmented_batches_and_CPU_CUDA_RNG_match_R0':True,'HyRANK_all_rng_and_actual_indices_match_R0':dataset=='hyrank','relation_parameters_BN_and_gradients_unchanged':True,'new_R0_training':False,'no_target_GT_scored':True,'seconds':time.monotonic()-start});print('SPECTRAL_TRAIN_COMPLETE',dataset,seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=s.DATASETS,required=True);p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);a=p.parse_args();main(a.dataset,a.seed)
