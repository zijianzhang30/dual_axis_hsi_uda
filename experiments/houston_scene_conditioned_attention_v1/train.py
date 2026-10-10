import argparse,hashlib,json,time
import numpy as np,torch
import torch.nn.functional as F
import study as s
from mechanism import install,base_hash,refresh

def main(arm):
 s.verify();assert json.loads((s.OUT/'preflight.json').read_text())['passed'];torch.set_num_threads(2);c=s.env();m,ls,helper,classes,ba=c.setup('houston','C',2100);source,val,target,ev=ls;cfg=json.loads((s.baseline()/'config.json').read_text());ia=install(m,arm);assert base_hash(m)==cfg['initial_full_model_sha256'] and ba['indices_sha256']==cfg['indices_sha256'];m.cuda();att=m.blocks[0].attn;target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels);opt=torch.optim.SGD(m.parameters(),lr=.01);assert {id(p) for p in m.parameters()}=={id(p) for g in opt.param_groups for p in g['params']}
 folder=s.OUT/(arm+'_2100');folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();config={'arm':arm,'seed':2100,'plan':s.PLAN,'base_audit':ba,'install_audit':ia,'initial_full_hash':c.state_hash(m),'parameters':384251};s.dump(folder/'config.json',config);ref=[json.loads(x) for x in (s.baseline()/'history.jsonl').read_text().splitlines()];cps=[];start=time.monotonic();calls={};hooks=[]
 for name,mod in [('block%d'%i,b) for i,b in enumerate(m.blocks)]+[('classifier',m.nn1)]:
  def make(n):
   def h(mod,args,out):calls[n]=calls.get(n,0)+1
   return h
  hooks.append(mod.register_forward_hook(make(name)))
 with (folder/'history.jsonl').open('x') as f:
  for epoch in range(1,101):
   m.train();att.scene_summary_mode='batch';calls.clear();bh=hashlib.sha256();rh=hashlib.sha256();steps=count=correct=0;loss_sum=0;first=None
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for a in (x,y,t):bh.update(a.numpy().tobytes())
    rh.update(torch.get_rng_state().numpy().tobytes());rh.update(torch.cuda.get_rng_state().numpy().tobytes());x,y,t=x.cuda(),y.cuda(),t.cuda();opt.zero_grad(set_to_none=True);att.scene_log=steps==0;out=m(x,t);loss=F.cross_entropy(out[0],y);assert torch.isfinite(loss)
    if steps==0:
     cg=torch.autograd.grad(loss,att.last_context,retain_graph=True)[0];first={'attention':dict(att.last_diagnostics),'source_CE_context_gradient_norm':float(cg.norm()),'summary_values':att.last_context.detach().cpu().tolist()}
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);assert att.scene_modulator[-1].weight.grad is not None
    if steps==0:first.update({'modulator_output_gradient_norm':float(att.scene_modulator[-1].weight.grad.norm()),'modulator_hidden_gradient_norm':float(att.scene_modulator[0].weight.grad.norm()),'qkv_gradient_norm':float(att.qkv.weight.grad.norm())})
    opt.step();steps+=1;n=len(y);count+=n;correct+=int((out[0].argmax(1)==y).sum());loss_sum+=float(loss.detach())*n
   assert steps==18 and bh.hexdigest()==ref[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==ref[epoch-1]['original_rng_stream_sha256'],('pairing_failure_STOP',arm,epoch)
   bn={n:int(v.num_batches_tracked) for n,v in m.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))};assert all(v==18*epoch for v in bn.values());assert all(calls['block%d'%i]==18 for i in range(3)) and calls['classifier']==72
   r={'epoch':epoch,'steps':steps,'source_CE':loss_sum/count,'source_train_OA':100*correct/count,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'BN_counts':bn,'normal_forward_calls':dict(calls),'first_batch_diagnostics':first,'target_class_GT_accessed':False,'elapsed_seconds':time.monotonic()-start}
   if epoch==1 or epoch%10==0:
    att.scene_log=False;r['fixed_scene_summary']=refresh(m,source.dataset,target.dataset);r['source_val']=helper.source_evaluate(m,val,torch.device('cuda:0'));print(json.dumps(r),flush=True)
   if epoch%10==0:
    cp=folder/'checkpoints'/('epoch_%03d.pth'%epoch);torch.save({'model':m.state_dict(),'epoch':epoch,'config':config,'fixed_scene_summary':r['fixed_scene_summary']},cp);cps.append({'epoch':epoch,'path':str(cp),'sha256':s.sha(cp)})
   f.write(json.dumps(r)+'\n');f.flush()
 for h in hooks:h.remove()
 s.verify();s.dump(folder/'frozen_manifest.json',{'completed_epochs':100,'optimizer_steps':1800,'checkpoints':cps,'train_seconds':time.monotonic()-start,'source_val_final':r['source_val'],'all100_main_data_RNG_hashes_match_S0':True,'normal_BN_updates_Transformer_calls_unchanged':True,'extra_summary_forwards':'11x1024/128 interrupted preattention eval forwards; noBNupdates/privategenerator','target_GT_scored':False});print('SCENE_TRAIN_COMPLETE',arm,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=s.ARMS,required=True);main(p.parse_args().arm)
