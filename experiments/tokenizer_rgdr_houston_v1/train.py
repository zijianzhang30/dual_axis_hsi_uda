import argparse,json,time,hashlib
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
from common import OUT,BASE,environment,loaders,build,state_hash,sha,dump
from diagnostics import diagnostic,describe

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['P0','P1'],required=True);args=p.parse_args();args.dataset='houston';args.seed=2100;args.num_workers=4;args.dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/'
 folder=OUT/'houston'/f'{args.arm}_2100_e100';folder.mkdir(exist_ok=False);(folder/'checkpoints').mkdir();assert json.loads((OUT/'verification.json').read_text())['passed'];prov=json.loads((OUT/'provenance.json').read_text())
 for name,h in prov['study_sha256'].items():assert sha(OUT/name)==h
 for name,h in prov['snapshot_sha256'].items():assert sha(OUT/'snapshot'/name)==h
 torch.set_num_threads(2);env=environment();env[4](2100);source,val,target,target_eval=loaders(args);model,initial=build(2100,args.arm,'houston');model.cuda();opt=torch.optim.SGD(model.parameters(),lr=.01);mem=model.prototype_reasoning
 selected=np.linspace(0,len(target_eval.dataset)-1,256,dtype=int);diag=torch.stack([target_eval.dataset[int(i)][0] for i in selected]).cuda();diag_hash=hashlib.sha256(diag.cpu().numpy().tobytes()).hexdigest()
 config={'dataset':'houston','arm':args.arm,'seed':2100,'epochs':100,'patch':13,'batch':128,'lr':.01,'depth':3,'dropout':.1,'loss':'source-self CE only','BN':'unchanged Full Joint/momentum.19','primary':'fixed_epoch100','tokens':4,'dim':64,'heads':8,'top_M':8,'memory_EMA':.9,'residual':.1,'memory_update':'first source observation bootstraps,thenEMA;update after step from detached raw source T/W;both domains read previous memory','initial_original_model_sha256':initial,'initial_full_model_sha256':state_hash(model),'initial_scorer_sha256':state_hash(model.relation_scores),'initial_reasoning_sha256':state_hash(mem),'parameters':sum(p.numel() for p in model.parameters()),'memory_T_W_bytes':mem.M_T.numel()*4+mem.M_W.numel()*4,'all_memory_buffers_bytes':sum(b.numel()*b.element_size() for b in mem.buffers()),'samples':{n:len(l.dataset) for n,l in [('source',source),('val',val),('target',target),('target_eval',target_eval)]},'indices_sha256':{n:hashlib.sha256(l.dataset.indices.tobytes()).hexdigest() for n,l in [('source',source),('val',val),('target',target),('target_eval',target_eval)]},'fixed_unlabeled_diagnostic_input_sha256':diag_hash,'optimizer_excludes_memory_buffers':True,'code_sha256':prov['study_sha256'],'training_target_labels_used':False,'target_GT_mask_sampling':'unchanged native loader protocol; values excluded from loss/memory/selection'}
 ref=BASE/'houston/C_2100';rc=json.loads((ref/'config.json').read_text());reference=[json.loads(v) for v in (ref/'history.jsonl').read_text().splitlines()]
 for k in ['initial_original_model_sha256','initial_scorer_sha256','indices_sha256','samples','patch','batch','lr','depth','dropout','loss','BN','tokens','dim']:assert config[k]==rc[k],k
 assert config['parameters']==rc['parameters']+16448;dump(folder/'config.json',config);checkpoints=[];start=time.monotonic()
 with (folder/'history.jsonl').open('x') as f:
  for epoch in range(1,101):
   model.train();total=count=steps=0;batchhash=hashlib.sha256();rnghash=hashlib.sha256();gradient_sum=0.
   for (x,y),(t,_) in zip(source,target):
    if len(x)!=len(t):continue
    for a in [x,y,t]:batchhash.update(a.numpy().tobytes())
    rnghash.update(torch.get_rng_state().numpy().tobytes());rnghash.update(torch.cuda.get_rng_state().numpy().tobytes());x=x.cuda();y=y.cuda();t=t.cuda();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x,t)[0],y);assert torch.isfinite(loss);loss.backward()
    assert mem.M_T.grad is None and mem.M_W.grad is None;gradient_sum+=sum(float(p.grad.norm()) for p in mem.parameters() if p.grad is not None);opt.step();model.update_source_memory(y);total+=float(loss)*len(y);count+=len(y);steps+=1
   assert steps==18;row={'epoch':epoch,'steps':steps,'source_ce':total/count,'reasoning_gradient_norm_sum_mean':gradient_sum/steps,'batch_stream_sha256':batchhash.hexdigest(),'original_rng_stream_sha256':rnghash.hexdigest(),'memory_valid_classes':mem.valid.tolist(),'memory_class_updates':mem.updates.tolist()}
   for k in ['batch_stream_sha256','original_rng_stream_sha256']:assert row[k]==reference[epoch-1][k],(epoch,k)
   if epoch==1 or epoch%10==0:
    row['source_val']=env[6].source_evaluate(model,val,'cuda');before=state_hash(model);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();row['unlabeled_target_memory_diagnostic']=diagnostic(model,diag);assert state_hash(model)==before and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state());print(json.dumps({k:v for k,v in row.items() if k!='unlabeled_target_memory_diagnostic'}),flush=True)
   if epoch%10==0:
    path=folder/'checkpoints'/f'epoch_{epoch:03d}.pth';torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},path);checkpoints.append({'epoch':epoch,'path':str(path),'sha256':sha(path)})
   f.write(json.dumps(row)+'\n');f.flush()
 torch.cuda.synchronize();seconds=time.monotonic()-start;model.eval();mem.diagnostics=True;before=state_hash(model);logits=[];ids=[];sims=[];atts=[]
 with torch.inference_mode():
  for x,_ in target_eval:
   out,capture,last=model.target_only(x.cuda(),return_details=True);logits.append(out.cpu().numpy());ids.append(last['selected'].cpu().numpy());sims.append(last['similarity'].cpu().numpy());atts.append(last['attention'].cpu().numpy())
 mem.diagnostics=False;assert state_hash(model)==before;outputs=np.concatenate(logits);np.savez_compressed(folder/'predictions_final.npz',logits=outputs,predictions=outputs.argmax(1));np.savez_compressed(folder/'memory_selection_final.npz',selected=np.concatenate(ids),similarity=np.concatenate(sims),attention=np.concatenate(atts))
 manifest={'config':config,'checkpoints':checkpoints,'prediction':{'path':str(folder/'predictions_final.npz'),'sha256':sha(folder/'predictions_final.npz')},'selection':{'path':str(folder/'memory_selection_final.npz'),'sha256':sha(folder/'memory_selection_final.npz')},'train_seconds':seconds,'source_val_at_final':row['source_val'],'memory_valid_classes':mem.valid.tolist(),'memory_class_updates':mem.updates.tolist(),'memory_state_sha256':state_hash(mem),'target_labels_scored_during_training':False,'target_only_inference':True};dump(folder/'frozen_manifest.json',manifest);print('RGDR fixed100 predictions frozen',flush=True)
if __name__=='__main__':main()
