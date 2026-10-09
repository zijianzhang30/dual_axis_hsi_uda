import argparse,json,time,numpy as np,torch
from argparse import Namespace
from common import OUT,BASE,SEEDS,environment,loaders,build,sha,dump,state_hash
from scr import compatibility
from offline_metrics import selectivity,separability

def stats(v):
 a=np.asarray(v,float);return {'values':a.tolist(),'mean':float(a.mean()),'sample_std':float(a.std(ddof=1)) if len(a)>1 else None}
class Patches(torch.utils.data.Dataset):
 def __init__(self,cube,coordinates):self.cube=cube;self.coordinates=coordinates
 def __len__(self):return len(self.coordinates)
 def __getitem__(self,i):
  x,y=self.coordinates[i];return torch.from_numpy(np.array(self.cube[x-6:x+7,y-6:y+7],np.float32).transpose(2,0,1)).unsqueeze(0)
@torch.inference_mode()
def extract(model,loader,device):
 arrays={k:[] for k in ['logits','W','T','Z','S_sem']};last={};original=model._forward_semantic_tokens
 def capture(features):
  patch=model._relation_pending[0];tokens=original(features)
  w=model._scr_weights if hasattr(model,'scr_beta') else model.relation_scores(patch)
  last['W']=w.detach();last['T']=tokens.detach();last['S_sem']=compatibility(features)
  return tokens
 model._forward_semantic_tokens=capture
 try:
  for x in loader:
   x=x.to(device);out=model(x,x,return_feat_prob=True);last['logits']=out[1];last['Z']=out[3]
   for k in arrays:arrays[k].append(last[k].cpu().numpy())
 finally:model._forward_semantic_tokens=original
 return {k:np.concatenate(v) for k,v in arrays.items()}
def semantic_distribution(s,gt,coords,shape):
 labels=np.stack([gt[x-6:x+7,y-6:y+7].reshape(-1) for x,y in coords]);center=labels[:,84];same=labels==center[:,None];other=(labels>=0)&~same;same[:,84]=False;other[:,84]=False
 xy=coords-6;inside=(xy[:,0]>=6)&(xy[:,0]<shape[0]-6)&(xy[:,1]>=6)&(xy[:,1]<shape[1]-6)
 result={}
 for name,mask in [('same_class',same),('different_class',other)]:
  v=s[mask&inside[:,None]]
  result[name]={'n_neighbor_pairs':len(v),'mean':float(v.mean()),'std':float(v.std()),'quantiles':np.percentile(v,[0,5,25,50,75,95,100]).tolist(),'per_center_class':{}}
  for c in range(7):
   vc=s[mask&(inside&(center==c))[:,None]]
   result[name]['per_center_class'][str(c+1)]={'n':len(vc),'mean':float(vc.mean()) if len(vc) else None,'std':float(vc.std()) if len(vc) else None}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,choices=[100,200],required=True);epochs=p.parse_args().epochs;seeds=[2100] if epochs==100 else list(SEEDS)
 root=OUT/f'evaluation_e{epochs}';root.mkdir(exist_ok=False);env=environment();torch.set_num_threads(2);device=torch.device('cuda:0');manifests={}
 for seed in seeds:
  f=OUT/'houston'/f'K1_{seed}_e{epochs}';m=json.loads((f/'frozen_manifest.json').read_text());manifests[seed]=m
  h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];reference=[json.loads(x) for x in (BASE/'houston'/f'C_{seed}/history.jsonl').read_text().splitlines()];assert len(h)==epochs
  for row,ref in zip(h,reference):
   for key in ['batch_stream_sha256','original_rng_stream_sha256']:assert row[key]==ref[key]
  for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
 dump(root/'all_K1_predictions_frozen.json',{'epochs':epochs,'seeds':seeds,'all_pairing_and_hash_checks_passed':True,'target_best':False})
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=2100);env[4](2100);ls=loaders(args);sc=np.unique(np.concatenate([ls[0].dataset.indices,ls[1].dataset.indices]),axis=0);tc=ls[3].dataset.indices
 from utils.dataset import open_file
 gs=open_file(args.dataset_dir+'Houston13_7gt.mat')['map'].astype(int)-1;gt=open_file(args.dataset_dir+'Houston18_7gt.mat')['map'].astype(int)-1;shapes={'source':gs.shape,'target':gt.shape};gs=np.pad(gs,6,mode='reflect');gt=np.pad(gt,6,mode='reflect');sl=gs[sc[:,0],sc[:,1]];tl=gt[tc[:,0],tc[:,1]];assert np.array_equal(tl,np.load(BASE/'houston/evaluation_labels.npy'))
 dl={'source':torch.utils.data.DataLoader(Patches(ls[0].dataset.data,sc),batch_size=128,shuffle=False,num_workers=2),'target':torch.utils.data.DataLoader(Patches(ls[3].dataset.data,tc),batch_size=128,shuffle=False,num_workers=2)}
 runs={}
 for seed in seeds:
  for arm in ['K0','K1']:
   cp=(BASE/'houston'/f'C_{seed}/checkpoints'/f'epoch_{epochs:03d}.pth') if arm=='K0' else (OUT/'houston'/f'K1_{seed}_e{epochs}/checkpoints'/f'epoch_{epochs:03d}.pth')
   env[4](seed);model,_=build(seed,'C' if arm=='K0' else 'K1','houston');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.to(device);model.eval();before=state_hash(model)
   domain={name:extract(model,loader,device) for name,loader in dl.items()}
   folder=root/f'{arm}_{seed}';folder.mkdir(exist_ok=False)
   for dom,a in domain.items():
    for key,v in a.items():np.save(folder/f'{dom}_{key}.npy',v)
   a=domain['target'];metrics=env[6].metrics(tl,a['logits'].argmax(1),a['logits'],7);metrics['collapse']=max(metrics['prediction_shares'])>=.95;metrics['zero_recall_classes']=[i+1 for i,v in enumerate(metrics['per_class_percent']) if v==0]
   if arm=='K1':
    prediction=np.load(manifests[seed]['prediction']['path']);assert np.array_equal(a['logits'].argmax(1),prediction['predictions']);assert np.max(abs(a['logits']-prediction['logits']))<1e-6
    w=np.load(manifests[seed]['weights']['path'],mmap_mode='r');assert np.max(abs(a['W']-w))<1e-6
   spatial=selectivity(a['W'],gt,tc,shapes['target']);separation={stage:separability(domain['source'][stage],sl,a[stage],tl) for stage in ['T','Z']};s_dist=semantic_distribution(a['S_sem'],gt,tc,shapes['target'])
   history=[json.loads(x) for x in ((BASE/'houston'/f'C_{seed}/history.jsonl') if arm=='K0' else (OUT/'houston'/f'K1_{seed}_e{epochs}/history.jsonl')).read_text().splitlines()]
   beta=model.scr_beta.detach().cpu().tolist() if arm=='K1' else [0.]*4
   run={'arm':arm,'seed':seed,'epoch':epochs,'metrics':metrics,'beta':beta,'beta_abs':list(map(abs,beta)),'source_val':history[epochs-1]['source_val'],'selectivity':spatial,'class_separability':separation,'s_sem_distribution':s_dist,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(cp),'train_seconds':manifests[seed]['train_seconds'] if arm=='K1' else (json.loads((BASE/'houston'/f'C_{seed}/frozen_manifest.json').read_text())['train_seconds'] if epochs==200 else None)}
   assert before==state_hash(model);runs[f'{arm}_{seed}']=run;dump(folder/'diagnostics.json',run);del model,domain
 keys=['oa_percent','aa_percent','kappa'];paired={k:stats([runs[f'K1_{s}']['metrics'][k]-runs[f'K0_{s}']['metrics'][k] for s in seeds]) for k in keys};agg={arm:{k:stats([runs[f'{arm}_{s}']['metrics'][k] for s in seeds]) for k in keys} for arm in ['K0','K1']}
 mechanisms={}
 for key in ['all_no_reflection','boundary3x3']:
  mechanisms[key+'_sameclass_delta']=stats([runs[f'K1_{s}']['selectivity'][key]['macro']['conditional_same_class_mass_excluding_center']-runs[f'K0_{s}']['selectivity'][key]['macro']['conditional_same_class_mass_excluding_center'] for s in seeds])
 for stage in ['T','Z']:mechanisms[stage+'_target_macro_margin_delta']=stats([runs[f'K1_{s}']['class_separability'][stage]['target']['macro_margin_LOO']-runs[f'K0_{s}']['class_separability'][stage]['target']['macro_margin_LOO'] for s in seeds])
 if epochs==100:
  metric_ok=max(paired['oa_percent']['mean'],paired['aa_percent']['mean'])>=.5 and min(paired['oa_percent']['mean'],paired['aa_percent']['mean'])>=-.5
  beta_ok=max(runs['K1_2100']['beta_abs'])>=.001
  selection_ok=mechanisms['all_no_reflection_sameclass_delta']['mean']>=.001 and mechanisms['boundary3x3_sameclass_delta']['mean']>0
  margin_ok=max(mechanisms['T_target_macro_margin_delta']['mean'],mechanisms['Z_target_macro_margin_delta']['mean'])>0
  collapse_ok=not runs['K1_2100']['metrics']['collapse'] and set(runs['K1_2100']['metrics']['zero_recall_classes'])<=set(runs['K0_2100']['metrics']['zero_recall_classes'])
  decision={'continue_stage2':bool(metric_ok and beta_ok and selection_ok and margin_ok and collapse_ok),'metric_signal':metric_ok,'beta_abs_ge_point001':beta_ok,'selectivity_all_ge_point001_and_boundary_positive':selection_ok,'T_or_Z_margin_improved':margin_ok,'no_collapse_or_new_zero_recall':collapse_ok,'no_auto_tuning':True}
 else:decision={'no_automatic_Pavia_or_new_variants':True}
 result={'fixed_epoch':epochs,'aggregate':agg,'paired_K1_minus_C':paired,'mechanism_paired_deltas':mechanisms,'runs':runs,'decision':decision,'parameters_added':4,'compute_extra_per_patch':{'channel_norm_squares':10816,'semantic_dot_MACs':10816,'bias_multiply_add_scalar_ops':1352},'limitations':'GT neighborcoverage incomplete;macroperclass;centerexcluded;reflectpatchesexcluded;targetclass4only22. CPU/GPU runtimeconfoundedbysharedjobs.'};dump(root/'summary.json',result)
 lines=[f'SCR Houston fixedepoch{epochs};baselineC SAME seed/endpoints','Aggregate '+json.dumps(agg),'K1-C '+json.dumps(paired),'Mechanism paired deltas '+json.dumps(mechanisms),'Decision '+json.dumps(decision)]
 for name,run in runs.items():lines.append(name+' '+json.dumps(run))
 lines+=['Params+4,total378015. CosinefromactualvalueV,stopgrad;noauxloss/extraCNN/Transformer.','Incremental semantic cosine10816MACs plusnorm10816squares,1352biasmultiply/add scalarops/patch. Timingnotcontrolledbenchmark.']
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
