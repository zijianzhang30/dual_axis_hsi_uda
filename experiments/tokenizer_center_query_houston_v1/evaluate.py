import json,numpy as np,torch,time
from common import OUT,BASE,environment,build,loaders,dump,sha
from argparse import Namespace

def main():
 torch.set_num_threads(2);env=environment();root=OUT/'evaluation_e100';root.mkdir(exist_ok=False);manifests={};histories={};seed=2100
 for arm in ['Q0','Q1']:
  f=OUT/'houston'/f'{arm}_2100_e100';m=json.loads((f/'frozen_manifest.json').read_text());manifests[arm]=m;h=[json.loads(x) for x in (f/'history.jsonl').read_text().splitlines()];assert len(h)==100;histories[arm]=h
  for item in [*m['checkpoints'],m['prediction'],m['weights'],m['tokens']]:assert sha(item['path'])==item['sha256']
 for a,b in zip(histories['Q0'],histories['Q1']):
  for k in ['batch_stream_sha256','original_rng_stream_sha256']:assert a[k]==b[k]
 assert manifests['Q0']['config']['initial_full_model_sha256']==manifests['Q1']['config']['initial_full_model_sha256']
 dump(root/'all_requested_predictions_frozen.json',{'Q0_Q1_fixed100_frozen':True,'init_and_100epoch_data_augmentation_CPU_CUDA_RNG_pairing_verified':True,'no_target_best':True})
 labels=np.load(BASE/'houston/evaluation_labels.npy');runs={}
 for arm in ['Q0','Q1']:
  m=manifests[arm];a=np.load(m['prediction']['path']);metric=env[6].metrics(labels,a['predictions'],a['logits'],7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[i+1 for i,v in enumerate(metric['per_class_percent']) if v==0]
  runs[arm]={'arm':arm,'seed':seed,'epoch':100,'metrics':metric,'parameters':m['config']['parameters'],'train_seconds':m['train_seconds'],'source_val':m['source_val_at_final']}
 # Independent eval C at same100 endpoint; baseline3layer not architecture-matched.
 args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=4,seed=seed);env[4](seed);loader=loaders(args)[3];model,_=build(seed,'C','houston');cp=BASE/'houston/C_2100/checkpoints/epoch_100.pth';model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);device=torch.device('cuda:0');model.to(device);model.eval();outputs=[]
 with torch.inference_mode():
  for x,_ in loader:x=x.to(device);outputs.append(model(x,x)[1].cpu().numpy())
 logits=np.concatenate(outputs);np.savez_compressed(root/'C_2100_e100_predictions.npz',logits=logits,predictions=logits.argmax(1));metric=env[6].metrics(labels,logits.argmax(1),logits,7);metric['collapse']=max(metric['prediction_shares'])>=.95;metric['zero_recall_classes']=[i+1 for i,v in enumerate(metric['per_class_percent']) if v==0]
 ch=[json.loads(x) for x in (BASE/'houston/C_2100/history.jsonl').read_text().splitlines()];runs['C']={'arm':'C','seed':seed,'epoch':100,'metrics':metric,'parameters':378011,'train_seconds':None,'source_val':ch[99]['source_val'],'checkpoint_sha256':sha(cp)}
 keys=['oa_percent','aa_percent','kappa'];paired={f'{a}-{b}':{k:runs[a]['metrics'][k]-runs[b]['metrics'][k] for k in keys} for a,b in [('Q1','Q0'),('Q1','C'),('Q0','C')]}
 gate=paired['Q1-Q0'];flag=max(gate['oa_percent'],gate['aa_percent'])>=.5 and min(gate['oa_percent'],gate['aa_percent'])>=-.5 and not runs['Q1']['metrics']['collapse']
 result={'dataset':'houston','fixed_epoch100':True,'seed':seed,'runs':runs,'paired_deltas':paired,'primary_contrast':'Q1-Q0,parameter/depth/initialization/data/RNGmatched','architecture_reference':'Q1-C differentdepth,parameters,anddropoutstreams;data/augmentationpaired','candidate_signal':flag,'no_auto_additional_runs':True,'Q1_residual_center_path_caveat':'Q1 includesVc inquery residual aswellasQprojection; gainalone cannot establish relation-context reasoning ratherthancenterfeaturebypass. Center-onlyfollowup onlyifexplicitlyrequested.'}
 dump(root/'summary.json',result)
 lines=['Houston Center Query controlled exploratory,seed2100,fixed100','Q0 learnedquery;Q1 samequery+actualCNNcenterVc. NativeC tokenizerunchanged. Bothoneexistingblockreused,crossattention8heads,64dim,4contexts,sameFFN/Linearclassifier.']
 for arm,r in runs.items():lines.append(arm+' '+json.dumps(r))
 lines+=['Paired deltas '+json.dumps(paired),'PrimaryQ1-Q0;C3layerreferenceisnotmatchedreadout.','Candidate classification signal '+str(flag),'Single-seedexploratoryonly, noformalstatisticalclaim orautomaticStage2/center-only/module.','Centerquery residual directly carriesVc;queryeffectandcenterbypass notyetisolated.']
 (root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps(paired),flush=True)
if __name__=='__main__':main()
