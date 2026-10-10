import argparse,json,time,hashlib
import numpy as np,torch
import torch.nn.functional as F
import study as s
from expert import build

@torch.inference_mode()
def source_evaluate(m,e,loader):
 m.eval();e.eval();n=0;cr=cs=0;lr=ls=0.
 for x,y in loader:
  x,y=x.cuda(),y.cuda();z=m(x,x)[1];v=e(x[:,0,:,6,6].detach());n+=len(y);cr+=int((z.argmax(1)==y).sum());cs+=int((v.argmax(1)==y).sum());lr+=float(F.cross_entropy(z,y))*len(y);ls+=float(F.cross_entropy(v,y))*len(y)
 return {'relation':{'oa_percent':cr/n*100,'loss':lr/n},'spectral':{'oa_percent':cs/n*100,'loss':ls/n}}
def main(seed):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);torch.cuda.set_device(0);c=s.environment();m,loaders,helper,n,audit=c.setup('houston','C',seed);source,val,target,ev=loaders;e=build(seed);m.cuda();e.cuda();ropt=torch.optim.SGD(m.parameters(),lr=.01);sopt=torch.optim.SGD(e.parameters(),lr=.01);assert not {id(p) for g in ropt.param_groups for p in g['params']} & {id(p) for g in sopt.param_groups for p in g['params']}
 for loader in (target,ev):loader.dataset.label=np.zeros_like(loader.dataset.label);loader.dataset.labels=np.zeros_like(loader.dataset.labels)
 f=s.OUT/('seed_%d'%seed);f.mkdir(exist_ok=False);(f/'checkpoints').mkdir();config={'seed':seed,'plan':s.PLAN,'loader_and_initial_audit':audit,'spectral_initial_hash':c.state_hash(e),'parameters':{'relation':378011,'spectral':3687},'separate_source_CE_optimizers':True};s.dump(f/'config.json',config);ref=[json.loads(v) for v in (s.baseline(seed)/'history.jsonl').read_text().splitlines()];calls={};handles=[]
 for name,mod in [('block_%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
  def hook(n):
   def h(mod,args,out):calls[n]=calls.get(n,0)+1
   return h
  handles.append(mod.register_forward_hook(hook(name)))
 start=time.monotonic();records=[];same100=None
 with (f/'history.jsonl').open('x') as stream:
  for epoch in range(1,201):
   m.train();e.train();calls.clear();bh=hashlib.sha256();rh=hashlib.sha256();steps=count=cr=cs=0;lr=ls=0.;diag=None
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bh.update(v.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes());x,y,t=x.cuda(),y.cuda(),t.cuda();ropt.zero_grad(set_to_none=True);sopt.zero_grad(set_to_none=True);z=m(x,t)[0];rl=F.cross_entropy(z,y);assert torch.isfinite(rl);rl.backward();ropt.step()
    cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();grads={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()} if steps==0 else None
    v=e(x[:,0,:,6,6].detach());sl=F.cross_entropy(v,y);assert torch.isfinite(sl);sl.backward();sopt.step();assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
    if steps==0:
     assert all((g is None and dict(m.named_parameters())[n].grad is None) or (g is not None and torch.equal(g,dict(m.named_parameters())[n].grad)) for n,g in grads.items());diag={'spectral_linear_grad_norm':float(e[1].weight.grad.norm()),'spectral_backward_keeps_relation_gradients_bitwise':True}
    steps+=1;count+=len(y);cr+=int((z.argmax(1)==y).sum());cs+=int((v.argmax(1)==y).sum());lr+=float(rl.detach())*len(y);ls+=float(sl.detach())*len(y)
   assert steps==18 and bh.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('source_data_rng_mismatch',seed,epoch)
   bn={n:int(v.num_batches_tracked) for n,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==18*epoch for v in bn.values());assert calls=={'block_0':18,'block_1':18,'block_2':18,'classifier':72}
   row={'seed':seed,'epoch':epoch,'steps':18,'relation_CE':lr/count,'spectral_CE':ls/count,'relation_source_OA':cr/count*100,'spectral_source_OA':cs/count*100,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'BN_counts':bn,'main_forward_calls':dict(calls),'gradient_isolation':diag,'elapsed_seconds':time.monotonic()-start}
   if epoch==1 or epoch%10==0:
    row['source_val']=source_evaluate(m,e,val);assert row['source_val']['relation']==ref[epoch-1]['source_val'];print(json.dumps(row),flush=True)
   if epoch%10==0:
    archived=torch.load(s.baseline(seed)/'checkpoints'/('epoch_%03d.pth'%epoch),map_location='cpu',weights_only=True)['model'];assert all(torch.equal(v.cpu(),archived[k]) for k,v in m.state_dict().items()),('relation_checkpoint_mismatch',seed,epoch)
    cp=f/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'relation':m.state_dict(),'spectral':e.state_dict(),'epoch':epoch,'config':config},cp);records.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp),'relation_bitwise_matches_archived_C':True})
    if seed==2100 and epoch==100:
     prev=torch.load(s.ROOT/'results/houston_relation_identity_dual_evidence_v1/spectral_epoch_100.pth',map_location='cpu',weights_only=True)['model'];assert all(torch.equal(v.cpu(),prev[k]) for k,v in e.state_dict().items());same100=True
   stream.write(json.dumps(row)+'\n');stream.flush()
 for h in handles:h.remove()
 s.verify();s.dump(f/'TRAINING_COMPLETE_FROZEN.json',{'seed':seed,'epochs':200,'updates':3600,'checkpoints':records,'all200_data_RNG_BN_and_calls_match_original_C':True,'all20_relation_checkpoints_bitwise_match_original_C':True,'spectral2100_epoch100_bitwise_matches_previous':same100,'no_target_GT_scored':True,'train_seconds':time.monotonic()-start});print('TRAIN_COMPLETE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=s.SEEDS,required=True);main(p.parse_args().seed)
