import argparse,json,time,hashlib
import numpy as np,torch
import torch.nn.functional as F
import study as s
import mechanism as a

def main(arm):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);torch.cuda.set_device(0);c=s.env();m,ls,helper,classes,audit=c.setup('houston','C',2100);source,val,target,ev=ls;old=json.loads((s.baseline()/'config.json').read_text());assert audit['complete_initial_hash']==old['initial_full_model_sha256'];target.dataset.labels=np.zeros_like(target.dataset.labels);target.dataset.label=np.zeros_like(target.dataset.label);source=a.wrap(source,s.OUT/'data/source_raw.npy');target=a.wrap(target,s.OUT/'data/target_raw.npy');m.cuda();d=a.decoder().cuda();assert sum(p.numel() for p in d.parameters())==9872;opt=torch.optim.SGD([*m.parameters(),*d.parameters()],lr=.01);assert {id(p) for p in [*m.parameters(),*d.parameters()]}=={id(p) for g in opt.param_groups for p in g['params']}
 calibration=json.loads((s.OUT/'source_only_calibration.json').read_text())[arm];lambda_aux=calibration['lambda_fixed'];q=np.load(s.OUT/('calibration_%s.npz'%arm));mean=torch.from_numpy(q['mean']).cuda();std=torch.from_numpy(q['std']).cuda();maskgen=torch.Generator().manual_seed(2100+157000);folder=s.OUT/(arm+'_2100');folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();config={'plan':s.PLAN,'arm':arm,'base_audit':audit,'calibration':calibration,'initial_main_hash':c.state_hash(m),'initial_decoder_hash':c.state_hash(d),'training_parameters':387883,'inference_parameters':378011,'main_initial_hash_matches_E0':True};s.dump(folder/'config.json',config);ref=[json.loads(x) for x in (s.baseline()/'history.jsonl').read_text().splitlines()];rows=[];cps=[];calls={};hooks=[];start=time.monotonic();globalstep=0
 for name,mod in [(n,b) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]+[('block_%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
  def make(name):
   def h(mod,args,out):calls[name]=calls.get(name,0)+1
   return h
  hooks.append(mod.register_forward_hook(make(name)))
 ps=list(m.parameters());names=[n for n,p in m.named_parameters()];back=[i for i,n in enumerate(names) if not n.startswith('nn1.')]
 with (folder/'history.jsonl').open('x') as h:
  for epoch in range(1,101):
   m.train();d.train();calls.clear();bh=hashlib.sha256();rh=hashlib.sha256();rawhash=hashlib.sha256();maskhash=hashlib.sha256();steps=count=0;ce_sum=aux_sum=src_sum=tgt_sum=0.;first=None
   for (x,y,rx),(t,dummy,rt) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bh.update(v.numpy().tobytes())
    for v in (rx,rt):rawhash.update(v.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes());x,y,rx,t,rt=x.cuda(),y.cuda(),rx.cuda(),t.cuda(),rt.cuda();opt.zero_grad(set_to_none=True);snap=a.snapshots(m);out=m(x,t);ce=F.cross_entropy(out[0],y);post={n:b.clone() for n,b in m.named_buffers()};mask=a.mask_batch(2*len(x),maskgen,'cuda');maskhash.update(mask.cpu().numpy().tobytes());ms,mt=mask.chunk(2,0);aux,diag,_=a.auxiliary(m,d,rx,rt,x,t,ms,mt,snap,arm,mean,std,2100+170000+globalstep);assert all(torch.equal(v,dict(m.named_buffers())[n]) for n,v in post.items());loss=ce+lambda_aux*aux;assert torch.isfinite(loss)
    if steps==0:
     gc=torch.autograd.grad(ce,ps,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(aux,ps,retain_graph=True,allow_unused=True);nc=a.norm([gc[i] for i in back]);ng=a.norm([ga[i] for i in back]);dot=sum(float(gc[i].flatten()@ga[i].flatten()) for i in back if gc[i] is not None and ga[i] is not None);assert nc>0 and ng>0 and all(torch.isfinite(v).all() for v in ga if v is not None);first={'CE_backbone_norm':nc,'aux_backbone_norm':ng,'weighted_aux_over_CE':lambda_aux*ng/nc,'gradient_cosine':dot/(nc*ng),'sample':'actual first batch ofepoch,notlongtermtrajectorysummary'};assert all(ga[i] is None or ga[i].abs().max()==0 for i,n in enumerate(names) if n.startswith('nn1.'));del gc,ga
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in [*m.parameters(),*d.parameters()] if p.grad is not None);opt.step();n=len(y);steps+=1;globalstep+=1;count+=n;ce_sum+=float(ce.detach())*n;aux_sum+=float(aux.detach())*n;src_sum+=diag['source_loss']*n;tgt_sum+=diag['target_loss']*n
   assert steps==18 and bh.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('native_main_random_or_data_stream_mismatch_STOP',arm,epoch)
   bn={n:int(b.num_batches_tracked) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==18*epoch for v in bn.values());assert all(calls['block_%d'%i]==36 for i in range(3)) and calls['classifier']==144 and all(calls[n]==36 for n in bn)
   row={'epoch':epoch,'steps':steps,'source_CE':ce_sum/count,'aux_Huber_source_target_mean':aux_sum/count,'source_aux_Huber':src_sum/count,'target_aux_Huber':tgt_sum/count,'weighted_aux':lambda_aux*aux_sum/count,'lambda_fixed':lambda_aux,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'raw_companion_sha256':rawhash.hexdigest(),'mask_sha256':maskhash.hexdigest(),'BN_updates':bn,'total_forward_calls':dict(calls),'gradient_diagnostic':first,'target_GT_scored':False,'elapsed_seconds':time.monotonic()-start}
   if epoch==1 or epoch%10==0:row['source_val']=helper.source_evaluate(m,val,torch.device('cuda:0'));print(json.dumps(row),flush=True)
   if epoch%10==0:
    cp=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'decoder':d.state_dict(),'epoch':epoch,'config':config},cp);cps.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp)})
   h.write(json.dumps(row)+'\n');h.flush()
 for v in hooks:v.remove()
 s.verify();s.dump(folder/'frozen_manifest.json',{'arm':arm,'completed_epochs':100,'optimizer_steps':1800,'normal_main_forwards':1800,'auxiliary_forwards':1800,'BN_running_updates':1800,'checkpoints':cps,'train_seconds':time.monotonic()-start,'main_initial_and_data_rng_match_E0':True,'target_GT_used':False,'source_val_final':row['source_val']});print('COMPLETION_TRAIN_COMPLETE',arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);main(p.parse_args().arm)
