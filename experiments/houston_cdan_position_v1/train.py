import argparse,hashlib,json,time,numpy as np,torch
import torch.nn.functional as F
import study as s,cdan_loss as a

def norm(g):return sum(float(x.square().sum()) for x in g if x is not None)**.5

def main(arm):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);c=s.environment();m,ls,helper,classes,audit=c.setup('houston','C',2100);source,val,target,ev=ls
 target.dataset.labels=np.zeros_like(target.dataset.labels);target.dataset.label=np.zeros_like(target.dataset.label);old=json.loads((s.baseline('CE')/'config.json').read_text());assert audit['complete_initial_hash']==old['initial_full_model_sha256'] and audit['indices_sha256']==old['indices_sha256']
 m=m.cuda();d=a.build_discriminator(2100).cuda();dhash=c.state_hash(d);bucket,clear,remove=a.capture(m);opt=torch.optim.SGD(m.parameters(),lr=.01);dopt=torch.optim.SGD(d.parameters(),lr=.01);ps=list(m.parameters());dp=list(d.parameters());names=[n for n,p in m.named_parameters()]
 folder=s.OUT/('CDAN_'+arm+'_2100');folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();config={'plan':s.PLAN,'arm':arm,'seed':2100,'base_audit':audit,'parameters_main':sum(p.numel() for p in ps),'parameters_critic':sum(p.numel() for p in dp),'critic_initial_sha256':dhash};s.dump(folder/'config.json',config)
 refs=[json.loads(x) for x in (s.baseline('CE')/'history.jsonl').read_text().splitlines()];start=time.monotonic();checkpoints=[];step=0;calls={}
 hooks=[]
 for name,mod in [('Transformer_%d'%i,b) for i,b in enumerate(m.blocks)]+[('critic',d),('classifier',m.nn1)]:
  def make(name):
   def cb(mod,inp,out):calls[name]=calls.get(name,0)+1
   return cb
  hooks.append(mod.register_forward_hook(make(name)))
 with (folder/'history.jsonl').open('x') as hist:
  for epoch in range(1,101):
   m.train();d.train();calls.clear();bh=hashlib.sha256();rh=hashlib.sha256();sums={'CE':0.,'domain_BCE':0.,'domain_accuracy':0.,'feature_norm':0.,'feature_variance':0.,'target_entropy':0.};nsteps=0;grad=None;coeffs=[]
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for v in (x,y,t):bh.update(v.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes());x,y,t=x.cuda(),y.cuda(),t.cuda();opt.zero_grad(set_to_none=True);dopt.zero_grad(set_to_none=True);clear();out=m(x,t);assert len(bucket['T'])==2 and len(bucket['Z'])==4;ce=F.cross_entropy(out[0],y);coeff=a.coefficient(step);dom,di=a.loss(d,tuple(bucket[arm][:2]),out[:2],coeff);loss=ce+dom;assert torch.isfinite(loss)
    if nsteps==0:
     g=torch.autograd.grad(dom,ps,retain_graph=True,allow_unused=True);gc=torch.autograd.grad(ce,ps,retain_graph=True,allow_unused=True);gd=torch.autograd.grad(dom,dp,retain_graph=True);groups={}
     for prefix in ('conv3d_features.','conv2d_features.','relation_scores.','blocks.','nn1.'):
      ids=[i for i,n in enumerate(names) if n.startswith(prefix)];groups[prefix]={'domain_GRL_gradient_norm':norm([g[i] for i in ids]),'CE_gradient_norm':norm([gc[i] for i in ids])}
     assert groups['nn1.']['domain_GRL_gradient_norm']==0 and (groups['blocks.']['domain_GRL_gradient_norm']==0 if arm=='T' else (coeff==0 or groups['blocks.']['domain_GRL_gradient_norm']>0))
     grad={'sample':'first actual batch same forward','coefficient':coeff,'CE_norm':norm(gc),'domain_GRL_main_norm':norm(g),'domain_over_CE':norm(g)/max(norm(gc),1e-12),'critic_norm':norm(gd),'by_module':groups};del g,gc,gd
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in ps+dp if p.grad is not None);opt.step();dopt.step();step+=1;nsteps+=1;coeffs.append(coeff)
    pp=out[1].detach().softmax(1);sums['CE']+=float(ce.detach());sums['domain_BCE']+=float(dom.detach());sums['domain_accuracy']+=di['domain_accuracy'];sums['feature_norm']+=di['features_norm'];sums['feature_variance']+=di['features_channel_variance'];sums['target_entropy']+=float(-(pp*pp.clamp_min(1e-12).log()).sum(1).mean())
   assert nsteps==18 and step==epoch*18;assert bh.hexdigest()==refs[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==refs[epoch-1]['original_rng_stream_sha256'],('data/RNG mismatch',arm,epoch)
   bn={n:int(b.num_batches_tracked) for n,b in m.named_modules() if isinstance(b,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==step for v in bn.values());assert calls['critic']==18 and all(calls['Transformer_%d'%i]==18 for i in range(3)) and calls['classifier']==72
   row={'epoch':epoch,'steps':18,'global_step':step,'GRL_first':coeffs[0],'GRL_last':coeffs[-1],**{k:v/18 for k,v in sums.items()},'gradient':grad,'BN_counts':bn,'forward_calls':dict(calls),'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'target_GT_scored':False}
   if epoch==1 or epoch%10==0:
    row['source_val']=helper.source_evaluate(m,val,torch.device('cuda:0'));row['elapsed_seconds']=time.monotonic()-start;path=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'discriminator':d.state_dict(),'epoch':epoch,'global_step':step,'config':config},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':s.sha(path)});print('CHECKPOINT',arm,epoch,'CE',row['CE'],'domain',row['domain_BCE'],'sourceval',row['source_val'],flush=True)
   hist.write(json.dumps(row)+'\n');hist.flush()
 remove()
 for h in hooks:h.remove()
 s.dump(folder/'frozen_manifest.json',{'completed_epochs':100,'global_step':step,'train_seconds':time.monotonic()-start,'checkpoints':checkpoints,'target_GT_not_scored':True});print('TRAIN_COMPLETE',arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('Z','T'),required=True);main(p.parse_args().arm)
