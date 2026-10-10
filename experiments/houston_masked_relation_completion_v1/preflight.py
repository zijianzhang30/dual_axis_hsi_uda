import json,hashlib,copy
import numpy as np,torch
import torch.nn.functional as F
import study as s
import mechanism as a

def grads(m,loss):
 ns=[(n,p) for n,p in m.named_parameters() if not n.startswith('nn1.')];g=torch.autograd.grad(loss,[p for n,p in ns],retain_graph=True,allow_unused=True);return ns,g

def main():
 s.verify();torch.set_num_threads(2);torch.cuda.set_device(0);c=s.env();m,ls,helper,classes,ba=c.setup('houston','C',2100);source,val,target,ev=ls;target.dataset.label=np.zeros_like(target.dataset.label);target.dataset.labels=np.zeros_like(target.dataset.labels);old=json.loads((s.baseline()/'config.json').read_text());assert ba['complete_initial_hash']==old['initial_full_model_sha256'];initial={n:v.clone() for n,v in m.state_dict().items()};m.cuda().train();d=a.decoder().cuda();initial_d=c.state_hash(d);ns=[(n,p) for n,p in m.named_parameters() if not n.startswith('nn1.')]
 # Calibration uses only labeled source training budget; first16 instances of each source class.
 ds=source.dataset;ys=np.asarray(ds.labels);ids=np.concatenate([np.flatnonzero(ys==k)[:16] for k in range(7)]);assert len(ids)==112 and all((ys[ids]==k).sum()==16 for k in range(7));cube=np.load(s.OUT/'data/source_raw.npy',mmap_mode='r');raw=[];full=[]
 for i in ids:
  x,y=ds.indices[i];raw.append(np.array(cube[x-6:x+7,y-6:y+7],copy=True).transpose(2,0,1));full.append(np.array(ds.data[x-6:x+7,y-6:y+7],copy=True).transpose(2,0,1))
 raw=torch.from_numpy(np.stack(raw)).unsqueeze(1).cuda();full=torch.from_numpy(np.stack(full)).unsqueeze(1).cuda();yl=torch.from_numpy(ys[ids]).cuda();assert float((a.full_normalize(raw)-full).abs().max())<1e-6
 gen=torch.Generator().manual_seed(2100+158000);mask=a.mask_batch(112,gen,'cuda');cal={};calfiles={}
 for arm in s.ARMS:
  m.load_state_dict(initial);snap=a.snapshots(m)
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(2100+180000);torch.cuda.manual_seed(2100+180000);ce=F.cross_entropy(m(full,full)[0],yl)
  _,gc=grads(m,ce);nc=a.norm(gc);values=a.target_values(m,full,arm).flatten(2);sites=torch.arange(169,device='cuda')!=84;mean=values[:,:,sites].mean((0,2));std=values[:,:,sites].std((0,2),correction=0).clamp_min(1e-6);loss,diag,_=a.auxiliary(m,d,raw,raw,full,full,mask,mask,snap,arm,mean,std,2100+181000);_,ga=grads(m,loss);ng=a.norm(ga);assert nc>0 and ng>0 and np.isfinite(nc+ng);lam=.1*nc/ng
  for n,v in m.named_modules():
   if n in snap:assert int(v.num_batches_tracked)==1
  assert all(torch.equal(snap[n]['mean'],initial[n+'.running_mean'].cuda()) for n in snap)
  path=s.OUT/('calibration_%s.npz'%arm);np.savez_compressed(path,mean=mean.cpu().numpy(),std=std.cpu().numpy(),source_training_indices=ids,source_coords=ds.indices[ids],mask=mask.cpu().numpy());calfiles[str(path)]=s.sha(path)
  cal[arm]={'lambda_fixed':lam,'CE_backbone_gradient_norm':nc,'aux_backbone_gradient_norm':ng,'initial_weighted_ratio':lam*ng/nc,'source_only':True,'source_samples':112,'source_class_counts':[16]*7,'native_input_normalization_error':float((a.full_normalize(raw)-full).abs().max()),'source_target_duplicate_source_calibration':True,'target_input_or_labels_used':False,'mean_range':[float(mean.min()),float(mean.max())],'std_range':[float(std.min()),float(std.max())],'decoder_initial_hash':initial_d,'optimizer_steps':0,'path':str(path),'sha256':s.sha(path)}
  m.load_state_dict(initial);assert c.state_hash(m)==old['initial_full_model_sha256'];assert c.state_hash(d)==initial_d
 s.dump(s.OUT/'source_only_calibration.json',cal);calfiles[str(s.OUT/'source_only_calibration.json')]=s.sha(s.OUT/'source_only_calibration.json');s.dump(s.OUT/'calibration_manifest.json',calfiles)
 # Actual native data replay: wrapping only adds raw companion; keeps original augmented primary tensors.
 src=a.wrap(source,s.OUT/'data/source_raw.npy');tar=a.wrap(target,s.OUT/'data/target_raw.npy');bh=hashlib.sha256();first=None;arch=np.load(s.B/'houston_oracle_lmmd_counterfactual_v1/inputs_2100.npz');steps=0
 for (x,y,rx),(t,dummy,rt) in zip(src,tar):
  if len(x)!=len(t):continue
  assert not dummy.any()
  for v in (x,y,t):bh.update(v.numpy().tobytes())
  if steps<3:assert np.array_equal(x.numpy(),arch['source'][steps]) and np.array_equal(t.numpy(),arch['target'][steps]) and np.array_equal(y.numpy(),arch['source_labels'][steps])
  if first is None:first=(x.cuda(),y.cuda(),rx.cuda(),t.cuda(),rt.cuda())
  assert float((a.full_normalize(rx)-x).abs().max())<1e-6 and float((a.full_normalize(rt)-t).abs().max())<1e-6;steps+=1
 ref=json.loads((s.baseline()/'history.jsonl').read_text().splitlines()[0]);assert steps==18 and bh.hexdigest()==ref['batch_stream_sha256']
 x,y,rx,t,rt=first;g=torch.Generator().manual_seed(2100+157000);masks=a.mask_batch(256,g,'cuda');ms,mt=masks[:128],masks[128:];audits=[]
 # E0 reference step and E1/E2 zero-coefficient step use same model, source/target batch, RNG.
 reference=None;grad_reference=None;post_reference=None;post_bn_ref=None
 for arm in ('E0','E1','E2'):
  m.load_state_dict(initial);m.train();snap=a.snapshots(m);torch.manual_seed(992100);torch.cuda.manual_seed(992100);out=m(x,t);ce=F.cross_entropy(out[0],y);rng_before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone());bn_before={n:b.clone() for n,b in m.named_buffers()};opt=torch.optim.SGD([*m.parameters(),*d.parameters()],lr=.01);opt.zero_grad(set_to_none=True);loss=ce;aux=None
  if arm!='E0':
   q=np.load(s.OUT/('calibration_%s.npz'%arm));mean=torch.from_numpy(q['mean']).cuda();std=torch.from_numpy(q['std']).cuda();aux,diag,_=a.auxiliary(m,d,rx,rt,x,t,ms,mt,snap,arm,mean,std,992101);loss=ce+0.*aux;assert all(torch.equal(v,dict(m.named_buffers())[n]) for n,v in bn_before.items());assert torch.equal(rng_before[0],torch.get_rng_state()) and torch.equal(rng_before[1],torch.cuda.get_rng_state())
   groups={}
   gg=torch.autograd.grad(aux,list(m.parameters()),retain_graph=True,allow_unused=True);names=[n for n,p in m.named_parameters()]
   for prefix in ('conv3d_features.','conv2d_features.','relation_scores.','blocks.','nn1.'):
    z=a.norm([v for n,v in zip(names,gg) if n.startswith(prefix)]);groups[prefix]=z;assert z>0 if prefix!='nn1.' else z==0
   assert all(torch.isfinite(v).all() for v in gg if v is not None)
   gd=torch.autograd.grad(aux,list(d.parameters()),retain_graph=True,allow_unused=True);assert a.norm(gd)>0 and all(torch.isfinite(v).all() for v in gd if v is not None)
   _,_,target_predictions=a.auxiliary(m,d,rx,rt,x,t,ms,mt,snap,arm,mean,std,992103);target_only=a.score(target_predictions[1],a.target_values(m,t,arm),mt,mean,std);gt=torch.autograd.grad(target_only,list(m.parameters()),retain_graph=True,allow_unused=True);assert a.norm(gt)>0 and all(torch.isfinite(v).all() for v in gt if v is not None)
   # Hidden-value replacement must not affect any auxiliary input or predicted output.
   replaced=rx.clone();replaced=torch.where(ms[:,None,:,None,None],replaced*1.7+11.,replaced);assert torch.equal(a.visible_input(rx,ms),a.visible_input(replaced,ms));full_replaced=a.full_normalize(replaced)
   _,_,pa=a.auxiliary(m,d,rx,rt,x,t,ms,mt,snap,arm,mean,std,992102);_,_,pb=a.auxiliary(m,d,replaced,rt,full_replaced,t,ms,mt,snap,arm,mean,std,992102);leak_error=float((pa[0]-pb[0]).abs().max());assert leak_error<1e-6
  loss.backward();grad={n:p.grad.clone() if p.grad is not None else None for n,p in m.named_parameters()};assert all(torch.isfinite(v).all() for v in grad.values() if v is not None)
  if arm=='E0':reference=out[0].detach().clone();grad_reference=grad
  else:
   assert torch.equal(out[0],reference);err=max(float((v-grad_reference[n]).abs().max()) for n,v in grad.items() if v is not None);assert err<1e-6
  opt.step();post={n:v.clone() for n,v in m.state_dict().items()}
  if arm=='E0':post_reference=post
  else:assert all(torch.equal(v,post_reference[n]) for n,v in post.items())
  audits.append({'arm':arm,'zero_lambda_logits_and_one_step_parameter_BN_equivalence':True,'gradient_error':0. if arm=='E0' else err,'aux_gradient_by_module':None if arm=='E0' else groups,'hidden_replacement_prediction_max_error':None if arm=='E0' else leak_error,'snapshot_independent_and_livebuffers_unchanged':True,'main_RNG_preserved':True,'target_GT_never_used':True})
 m.load_state_dict(initial);s.verify();s.dump(s.OUT/'preflight.json',{'passed':True,'calibration':cal,'audits':audits,'native_first_epoch_augmented_batch_hash_exact':True,'extra_parameters_decoder':9872,'new_main_network_epochs':0,'targets_detached':True,'same_counted_sites_E1_E2':True});print('COMPLETION_PREFLIGHT_PASSED',json.dumps(cal),flush=True)
if __name__=='__main__':main()
