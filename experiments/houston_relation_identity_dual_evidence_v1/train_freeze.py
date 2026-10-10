import json,time,hashlib
import numpy as np,torch
import torch.nn.functional as F
import study as s
from expert import build

@torch.inference_mode()
def source_evaluate(expert,loader):
 expert.eval();n=correct=0;loss=0.
 for x,y in loader:
  x=x[:,0,:,6,6].detach().cuda();y=y.cuda();z=expert(x);n+=len(y);correct+=int((z.argmax(1)==y).sum());loss+=float(F.cross_entropy(z,y))*len(y)
 return {'OA':correct/n*100,'CE':loss/n}
def main():
 s.verify();torch.set_num_threads(2);torch.cuda.set_device(0);c=s.environment();relation,ls,helper,n,audit=c.setup('houston','C',s.SEED);source,val,target,ev=ls
 initial=c.state_hash(relation);relation.load_state_dict(torch.load(s.relation()/'checkpoints/epoch_100.pth',map_location='cpu',weights_only=True)['model']);relation.requires_grad_(False);relation.eval();frozen=c.state_hash(relation);expert=build().cuda();opt=torch.optim.SGD(expert.parameters(),lr=.01);assert not {id(p) for p in relation.parameters()} & {id(p) for g in opt.param_groups for p in g['params']};assert all(not p.requires_grad for p in relation.parameters());assert sum(p.numel() for p in relation.parameters())==378011
 target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels);ev.dataset.label=np.zeros_like(ev.dataset.label);ev.dataset.labels=np.zeros_like(ev.dataset.labels)
 trace=torch.load(s.relation()/'paired_rng.pt',map_location='cpu',weights_only=True);assert len(trace)==1800;reference=[json.loads(x) for x in (s.relation()/'history.jsonl').read_text().splitlines()];s.dump(s.OUT/'config.json',{'plan':s.PLAN,'base_initial_hash':initial,'frozen_relation_hash':frozen,'loader_audit':audit,'spectral_parameters':3687,'spectral_initial_hash':c.state_hash(expert),'optimizer_excludes_relation_parameters':True})
 # Disposable source-only gradient isolation probe; restore expert/global states.
 g=torch.Generator().manual_seed(921460);xs=torch.rand(8,48,generator=g).cuda();ys=(torch.arange(8)%7).cuda();before=c.state_hash(expert);cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();F.cross_entropy(expert(xs),ys).backward();grad=float(expert[1].weight.grad.norm());assert grad>0 and all(p.grad is None for p in relation.parameters()) and c.state_hash(relation)==frozen and c.state_hash(expert)==before;assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());opt.zero_grad(set_to_none=True);s.dump(s.OUT/'preflight.json',{'passed':True,'source_only_spectral_gradient_norm':grad,'relation_gradients_all_None':True,'relation_parameters_excluded_from_optimizer':True,'relation_hash_unchanged':True,'CPU_CUDA_RNG_unchanged':True,'disposable_probe_no_optimizer_update':True})
 start=time.monotonic();total=0
 with (s.OUT/'history.jsonl').open('x') as stream:
  for epoch in range(1,101):
   expert.train();bh=hashlib.sha256();rh=hashlib.sha256();steps=count=correct=0;loss_sum=0.;gn=None
   for (x,y),(t,dummy) in zip(source,target):
    if len(x)!=len(t):continue
    assert not dummy.any()
    for a in (x,y,t):bh.update(a.numpy().tobytes())
    cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone();r=trace[total];assert torch.equal(cpu,r['pre_cpu']) and torch.equal(gpu,r['pre_cuda']),('pre_rng_mismatch',epoch,steps);rh.update(cpu.numpy().tobytes());rh.update(gpu.numpy().tobytes());center=x[:,0,:,6,6].detach().cuda();y=y.cuda();opt.zero_grad(set_to_none=True);z=expert(center);loss=F.cross_entropy(z,y);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in relation.parameters());opt.step();assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());torch.set_rng_state(r['post_cpu']);torch.cuda.set_rng_state(r['post_cuda'])
    if steps==0:gn=float(expert[1].weight.grad.norm())
    total+=1;steps+=1;count+=len(y);correct+=int((z.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
   assert steps==18 and bh.hexdigest()==reference[epoch-1]['batch_stream_sha256'] and rh.hexdigest()==reference[epoch-1]['original_rng_stream_sha256'],('data_or_rng_mismatch',epoch)
   assert c.state_hash(relation)==frozen and not relation.training and all(p.grad is None for p in relation.parameters())
   row={'epoch':epoch,'steps':steps,'source_train_OA':correct/count*100,'source_CE':loss_sum/count,'encoder_grad_norm':gn,'batch_stream_sha256':bh.hexdigest(),'original_rng_stream_sha256':rh.hexdigest(),'relation_hash_unchanged':True,'relation_gradients_None':True,'relation_BN_updates':0,'elapsed_seconds':time.monotonic()-start}
   if epoch==1 or epoch%10==0:row['source_val']=source_evaluate(expert,val);print(json.dumps(row),flush=True)
   stream.write(json.dumps(row)+'\n');stream.flush()
 assert total==1800;s.verify();cp=s.OUT/'spectral_epoch_100.pth';torch.save({'model':expert.state_dict(),'epoch':100,'plan':s.PLAN},cp);s.dump(s.OUT/'training_complete.json',{'epochs':100,'updates':1800,'checkpoint_sha256':s.sha(cp),'relation_hash_before_after':frozen,'relation_BN_updates':0,'all100_source_aug_data_RNG_hashes_match_R0':True,'target_GT_not_scored':True,'train_seconds':time.monotonic()-start})
 # Fixed100 target inference only; all labels zero, frozen relation logits reused.
 expert.eval();ds=ev.dataset;loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(187100));logits=[];before=c.state_hash(expert)
 with torch.inference_mode():
  for x,dummy in loader:
   assert not dummy.any();logits.append(expert(x[:,0,:,6,6].cuda()).cpu().numpy())
 assert c.state_hash(expert)==before and c.state_hash(relation)==frozen;spec=np.concatenate(logits);old=np.load(s.relation()/'offline_predictions/epoch_100.npz');rel=old['logits'];coords=ds.indices;assert np.array_equal(coords,old['coordinates']) and rel.shape==spec.shape==(52901,7)
 rel=torch.from_numpy(rel.astype(np.float64));spec=torch.from_numpy(spec.astype(np.float64));lr=F.log_softmax(rel,1);ls=F.log_softmax(spec,1);lp2=torch.logaddexp(lr,ls)-np.log(2.);lp3=F.log_softmax(lr+ls,1)
 values={'R0':rel.numpy(),'R1':spec.numpy(),'R2':lp2.numpy(),'R3':lp3.numpy()};pred={k:v.argmax(1).astype(np.int64) for k,v in values.items()};assert np.array_equal(pred['R0'],old['predictions']);agree=pred['R0']==pred['R1'];rows=[]
 for k,z in values.items():
  path=s.OUT/(k+'_fixed100_frozen.npz');np.savez_compressed(path,logits=z,predictions=pred[k],coordinates=coords);rows.append({'router':k,'path':str(path),'sha256':s.sha(path),'logit_semantics':'normalized log probability' if k in ('R2','R3') else 'raw expert logits'})
 path=s.OUT/'expert_agreement_frozen.npz';np.savez_compressed(path,coordinates=coords,relation_prediction=pred['R0'],spectral_prediction=pred['R1'],agreement=agree,disagreement=~agree);s.dump(s.OUT/'ALL_TARGET_EVIDENCE_FROZEN_BEFORE_GT.json',{'records':rows,'agreement_path':str(path),'agreement_sha256':s.sha(path),'relation_checkpoint_sha256':s.sha(s.relation()/'checkpoints/epoch_100.pth'),'spectral_checkpoint_sha256':s.sha(cp),'source_only_training_complete':True,'target_GT_not_scored_in_this_study':True,'historical_R0_target_metrics_already_known':True,'fusion_rules_prelocked':True,'R0_R1_agree_count':int(agree.sum()),'R0_R1_disagree_count':int((~agree).sum())});s.verify();print('DUAL_EVIDENCE_TRAIN_AND_FREEZE_COMPLETE',flush=True)
if __name__=='__main__':main()
