import argparse,json,numpy as np,torch,time
from common import OUT,BASE,SEEDS,environment,loaders,build,sha,dump
from rpti import provenance

def stats(v):
 x=np.array(v,float);return {'values':x.tolist(),'mean':float(x.mean()),'sample_std':float(x.std(ddof=1)) if len(x)>1 else None}
def endpoint(model,loader,device,save_prediction=None):
 model.eval();logits=[];ents=[];rs=[]
 with torch.inference_mode():
  for x,_ in loader:
   x=x.to(device);out=model(x,x)[1];logits.append(out.cpu().numpy())
   e=[]
   for blk in model.blocks:
    a=blk.attn.attn;e.append((-(a*a.clamp_min(torch.finfo(a.dtype).tiny).log()).sum(-1)).mean(2).cpu().numpy())
   ents.append(np.stack(e,1));w=model.relation_scores(x);r=provenance(w);mask=~torch.eye(4,dtype=torch.bool,device=device);rs.append(r[:,mask].cpu().numpy())
 logits=np.concatenate(logits);ent=np.concatenate(ents);r=np.concatenate(rs)
 if save_prediction is not None:np.savez_compressed(save_prediction,logits=logits,predictions=logits.argmax(1))
 return logits,{'attention_entropy_nats_per_layer_head':ent.mean(0).tolist(),'R_offdiag_min':float(r.min()),'R_offdiag_mean':float(r.mean()),'R_offdiag_max':float(r.max()),'R_offdiag_std':float(r.std())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,choices=[100,200],required=True);args=p.parse_args();epochs=args.epochs;seeds=[2100] if epochs==100 else list(SEEDS)
 torch.set_num_threads(2);env=environment();device=torch.device('cuda:0');root=OUT/f'evaluation_e{epochs}';root.mkdir(exist_ok=False);manifests={}
 for seed in seeds:
  f=OUT/'houston'/f'J1_{seed}_e{epochs}';m=json.loads((f/'frozen_manifest.json').read_text());manifests[seed]=m
  h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(h)==epochs
  ref=[json.loads(x) for x in (BASE/'houston'/f'C_{seed}/history.jsonl').read_text().splitlines()]
  for a,b in zip(h,ref):
   for k in ['batch_stream_sha256','original_rng_stream_sha256']:assert a[k]==b[k]
  for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
 dump(root/'all_requested_predictions_frozen.json',{'J1_endpoint':epochs,'seeds':seeds,'pairing_audit_passed':True,'no_target_best':True})
 labels=np.load(BASE/'houston/evaluation_labels.npy');runs={}
 # Only predetermined endpoints are evaluated. No target-label use in training.
 for seed in seeds:
  la=__import__('argparse').Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=4,seed=seed)
  env[4](seed);loader=loaders(la)[3]
  for arm in ['C0','J1']:
   checkpoint=(BASE/'houston'/f'C_{seed}/checkpoints'/f'epoch_{epochs:03d}.pth') if arm=='C0' else (OUT/'houston'/f'J1_{seed}_e{epochs}/checkpoints'/f'epoch_{epochs:03d}.pth')
   model,_=build(seed,'C' if arm=='C0' else 'J1','houston');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.to(device)
   t=time.monotonic();logits,diag=endpoint(model,loader,device,root/f'{arm}_{seed}_predictions.npz');seconds=time.monotonic()-t
   if arm=='J1':
    frozen=np.load(manifests[seed]['prediction']['path']);assert np.array_equal(logits.argmax(1),frozen['predictions']);maxerr=float(np.max(abs(logits-frozen['logits'])));assert maxerr<1e-6,maxerr
   metric=env[6].metrics(labels,logits.argmax(1),logits,7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[i+1 for i,v in enumerate(metric['per_class_percent']) if v==0]
   history=[json.loads(x) for x in ((BASE/'houston'/f'C_{seed}/history.jsonl') if arm=='C0' else (OUT/'houston'/f'J1_{seed}_e{epochs}/history.jsonl')).read_text().splitlines()]
   beta=[b.attn.beta.detach().cpu().tolist() for b in model.blocks] if arm=='J1' else [[0.]*8]*3
   array=np.array(beta)
   run={'arm':arm,'seed':seed,'epoch':epochs,'metrics':metric,'beta':beta,'beta_positive_fraction':float((array>0).mean()),'beta_negative_fraction':float((array<0).mean()),'beta_abs_max':float(abs(array).max()),'diagnostics':diag,'source_val':history[epochs-1]['source_val'],'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(checkpoint),'inference_seconds':seconds}
   if arm=='J1':run['train_seconds']=manifests[seed]['train_seconds']
   elif epochs==200:run['train_seconds']=json.loads((BASE/'houston'/f'C_{seed}/frozen_manifest.json').read_text())['train_seconds']
   else:run['train_seconds']=None
   runs[f'{arm}_{seed}']=run;del model
 keys=['oa_percent','aa_percent','kappa'];paired={k:stats([runs[f'J1_{s}']['metrics'][k]-runs[f'C0_{s}']['metrics'][k] for s in seeds]) for k in keys}
 agg={arm:{k:stats([runs[f'{arm}_{s}']['metrics'][k] for s in seeds]) for k in keys} for arm in ['C0','J1']}
 if epochs==100:
  oa=paired['oa_percent']['mean'];aa=paired['aa_percent']['mean'];positive=max(oa,aa)>=.5 and min(oa,aa)>=-.5;beta_nonzero=runs['J1_2100']['beta_abs_max']>=.001;collapse=any(runs[k]['metrics']['collapse'] for k in runs if k.startswith('J1'))
  decision={'continue_stage2':bool(positive and beta_nonzero and not collapse),'rule':'OA orAA >=+.5pp; other>=-.5pp; maxabsbeta>=.001; no class collapse','clear_metric_signal':positive,'beta_away_from_zero':beta_nonzero,'no_auto_complex_modules':True}
 else:decision={'no_automatic_followup':True}
 result={'stage':1 if epochs==100 else 2,'fixed_epoch':epochs,'aggregate':agg,'paired_J1_minus_C':paired,'runs':runs,'decision':decision,'parameters_added':24,'parameters_J1':378035,'incremental_operation_counts_per_patch':{'R_dot_MACs':2704,'R_L2_norm_squares':676,'bias_multiply_add_scalar_ops':1200},'operation_notes':'R computed once per source/target patch; 3layers*8heads*5*5 multiply/add. Count uses dense implementation includingzeroCLS. Detach and caching do not add parameter networks. Timing confounded by sharedGPU loads.','last_layer_beta_has_no_CLS_classification_gradient_by_structure':True}
 dump(root/'summary.json',result)
 lines=[f'RPTI Houston Stage{result["stage"]},fixedepoch{epochs}','Paired baseline: original C checkpoint at SAME epoch, native eval pairing/pixels.']
 for name,run in runs.items():lines.append(name+' '+json.dumps(run))
 lines+=['Aggregate '+json.dumps(agg),'J1-C '+json.dumps(paired),'Decision '+json.dumps(decision),'Params overhead24. FLOPs convention:2perMAC;R dot5408FLOPs plusnorm,cosine,1200 biasarithmetic scalarops. No extraCNN/Transformer forward.','No best checkpoint, no loss change, no Pavia mutation.']
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
