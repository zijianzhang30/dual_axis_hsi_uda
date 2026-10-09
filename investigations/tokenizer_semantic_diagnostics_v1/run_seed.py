import sys
sys.dont_write_bytecode=True
import pathlib,json,numpy as np,torch,time,argparse
from argparse import Namespace
BASE=pathlib.Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tokenizer_mechanism_v2');OUT=pathlib.Path(__file__).parent
sys.path.insert(0,str(BASE));import common
LEVELS=[0.,.05,.10,.20,.40,.80]
class Patches(torch.utils.data.Dataset):
 def __init__(self,data,coords):self.data=data;self.coords=coords
 def __len__(self):return len(self.coords)
 def __getitem__(self,i):
  x,y=self.coords[i];return torch.from_numpy(np.array(self.data[x-6:x+7,y-6:y+7],np.float32).transpose(2,0,1)).unsqueeze(0)

def dump(p,d):pathlib.Path(p).write_text(json.dumps(d,indent=2)+'\n')
def shape_stats(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'std':float(a.std()) if len(a) else None}
def selectivity(w,gt,coords,shape):
 # GT onlyoffline; ignore undefinedneighbors ratherthan treating themaswrongclass.
 labels=np.stack([gt[x-6:x+7,y-6:y+7].reshape(-1) for x,y in coords]);center=labels[:,84];known=labels>=0;same=labels==center[:,None]
 known[:,84]=False;same[:,84]=False;knowncount=known.sum(1);samecount=same.sum(1)
 n=len(w);same_mass=(w*same[:,None,:]).sum(-1);known_mass=(w*known[:,None,:]).sum(-1);cond=np.divide(same_mass,known_mass,out=np.full_like(same_mass,np.nan),where=known_mass>1e-12).mean(1)
 uniform=np.divide(samecount,knowncount,out=np.full(n,np.nan),where=knowncount>0);excess=cond-uniform
 central=labels.reshape(n,13,13)[:,5:8,5:8].reshape(n,9);boundary=((central>=0)&(central!=center[:,None])).any(1);interior=(central==center[:,None]).all(1)
 full_patch_mixed=((labels>=0)&(labels!=center[:,None])).any(1)
 original=coords-6;no_reflection=(original[:,0]>=6)&(original[:,0]<shape[0]-6)&(original[:,1]>=6)&(original[:,1]<shape[1]-6)
 vals={'conditional_same_class_mass_excluding_center':cond,'excess_over_uniform_known_neighbors':excess,'center_weight':w[:,:,84].mean(1),'raw_same_class_mass_excluding_center':same_mass.mean(1),'known_neighbor_weight_mass_excluding_center':known_mass.mean(1),'known_neighbor_fraction':knowncount/168}
 result={}
 for name,mask in [('all_no_reflection',no_reflection),('boundary3x3',boundary&no_reflection),('interior3x3_fully_labeled',interior&no_reflection),('mixed13x13',full_patch_mixed&no_reflection)]:
  row={'patch_count':int(mask.sum()),'micro':{k:shape_stats(v[mask]) for k,v in vals.items()},'per_class':{}}
  for c in range(7):
   cm=mask&(center==c);row['per_class'][str(c+1)]={k:shape_stats(v[cm]) for k,v in vals.items()}
  row['macro']={k:float(np.mean([q[k]['mean'] for q in row['per_class'].values() if q[k]['n']])) if any(q[k]['n'] for q in row['per_class'].values()) else None for k in vals}
  result[name]=row
 return result

def separability(source,sl,target,tl):
 s=np.asarray(source,float).reshape(len(sl),-1);t=np.asarray(target,float).reshape(len(tl),-1)
 s/=np.maximum(np.linalg.norm(s,axis=1,keepdims=True),1e-30);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-30)
 domain={};centers={}
 for name,x,y in [('source',s,sl),('target',t,tl)]:
  sums=np.stack([x[y==c].sum(0) for c in range(7)]);counts=np.array([(y==c).sum() for c in range(7)]);assert (counts>1).all(),counts
  center=sums/np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-30);centers[name]=center
  sim=x@center.T;other=sim.copy();other[np.arange(len(x)),y]=-np.inf
  ownloo=sums[y]-x;ownloo/=np.maximum(np.linalg.norm(ownloo,axis=1,keepdims=True),1e-30);owncos=(x*ownloo).sum(1);margin=owncos-other.max(1)
  pair=1-center@center.T;np.fill_diagonal(pair,np.inf)
  cls={str(c+1):{'n':int(counts[c]),'within_class_cosine_distance_LOO':float((1-owncos[y==c]).mean()),'nearest_other_center_distance':float(pair[c].min()),'nearest_other_center_margin_LOO':float(margin[y==c].mean()),'positive_margin_fraction':float((margin[y==c]>0).mean())} for c in range(7)}
  domain[name]={'per_class':cls,'macro_within_class_cosine_distance':float(np.mean([v['within_class_cosine_distance_LOO'] for v in cls.values()])),'macro_nearest_center_distance':float(np.mean([v['nearest_other_center_distance'] for v in cls.values()])),'macro_margin_LOO':float(np.mean([v['nearest_other_center_margin_LOO'] for v in cls.values()])),'macro_positive_margin_fraction':float(np.mean([v['positive_margin_fraction'] for v in cls.values()])),'centroid_cosine_matrix':(center@center.T).tolist()}
 cross=centers['source']@centers['target'].T;diag=np.diag(cross);off=cross.copy();np.fill_diagonal(off,-np.inf);center_margin=diag-off.max(0)
 proto=t@centers['source'].T;wrong=proto.copy();wrong[np.arange(len(t)),tl]=-np.inf;samplemargin=proto[np.arange(len(t)),tl]-wrong.max(1);pred=proto.argmax(1)
 cls={str(c+1):{'n':int((tl==c).sum()),'matched_center_cosine':float(diag[c]),'matched_center_margin':float(center_margin[c]),'target_to_source_center_margin':float(samplemargin[tl==c].mean()),'source_prototype_target_recall':float((pred[tl==c]==c).mean())} for c in range(7)}
 domain['source_to_target']={'per_class':cls,'source_target_center_cosine_matrix':cross.tolist(),'macro_matched_center_cosine':float(diag.mean()),'macro_matched_center_margin':float(center_margin.mean()),'macro_target_to_source_center_margin':float(np.mean([v['target_to_source_center_margin'] for v in cls.values()])),'macro_source_prototype_target_recall':float(np.mean([v['source_prototype_target_recall'] for v in cls.values()]))}
 return domain
@torch.inference_mode()
def infer(model,loader,device,gain=None,offset=None,capture=False):
 logits=[];zs=[];ws=[];ts=[];orig=model._forward_semantic_tokens;last={}
 if capture:
  def token(v):
   last['W']=model.relation_scores(model._relation_pending[0]) if hasattr(model,'relation_scores') else model.conv_a(v).flatten(2).softmax(-1)
   z=orig(v);last['T']=z;return z
  model._forward_semantic_tokens=token
 try:
  for x in loader:
   x=x.to(device)
   if gain is not None:x=gain*x+offset
   out=model(x,x,return_feat_prob=True);logits.append(out[1].cpu().numpy());zs.append(out[3].cpu().numpy())
   if capture:ws.append(last['W'].cpu().numpy());ts.append(last['T'].cpu().numpy())
 finally:model._forward_semantic_tokens=orig
 return np.concatenate(logits),np.concatenate(zs),np.concatenate(ws) if ws else None,np.concatenate(ts) if ts else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[2100,2101,2102],required=True);seed=p.parse_args().seed
 torch.set_num_threads(2);env=common.environment();env[4](2100);args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',num_workers=0,seed=2100);ls=common.loaders(args)
 sourcecoords=np.unique(np.concatenate([ls[0].dataset.indices,ls[1].dataset.indices]),axis=0);targetcoords=ls[3].dataset.indices
 from utils.dataset import open_file
 gs=open_file(args.dataset_dir+'Houston13_7gt.mat')['map'].astype(int)-1;gt=open_file(args.dataset_dir+'Houston18_7gt.mat')['map'].astype(int)-1;shapes={'source':gs.shape,'target':gt.shape};gs=np.pad(gs,6,mode='reflect');gt=np.pad(gt,6,mode='reflect')
 sl=gs[sourcecoords[:,0],sourcecoords[:,1]];tl=gt[targetcoords[:,0],targetcoords[:,1]];assert np.array_equal(tl,np.load(BASE/'houston/evaluation_labels.npy'))
 loaders={'source':torch.utils.data.DataLoader(Patches(ls[0].dataset.data,sourcecoords),batch_size=128,num_workers=2,shuffle=False),'target':torch.utils.data.DataLoader(Patches(ls[3].dataset.data,targetcoords),batch_size=128,num_workers=2,shuffle=False)}
 folder=OUT/f'seed_{seed}';folder.mkdir(exist_ok=False);np.save(folder/'source_coordinates.npy',sourcecoords);np.save(folder/'target_coordinates.npy',targetcoords);np.save(folder/'source_labels_offline.npy',sl);np.save(folder/'target_labels_offline.npy',tl)
 vectors=np.load(OUT/'affine_vectors.npz');device=torch.device('cuda:0');results={}
 for arm in 'AC':
  env[4](seed);model,_=common.build(seed,arm,'houston');cp=BASE/'houston'/f'{arm}_{seed}/checkpoints/epoch_200.pth';cp_hash=common.sha(cp);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.to(device);model.eval();before=common.state_hash(model)
  curve=[];cleanZ=None
  for level in LEVELS:
   gain=torch.from_numpy(np.exp(level*vectors['gain_direction']).astype(np.float32)).to(device).reshape(1,1,48,1,1);offset=torch.from_numpy((level*.1*vectors['offset_direction']).astype(np.float32)).to(device).reshape(1,1,48,1,1)
   start=time.monotonic();logits,z,_,_=infer(model,loaders['target'],device,gain,offset);metrics=env[6].metrics(tl,logits.argmax(1),logits,7)
   np.savez_compressed(folder/f'{arm}_affine_{level:.2f}.npz',logits=logits,predictions=logits.argmax(1));curve.append({'strength':level,'metrics':metrics,'seconds':time.monotonic()-start})
   if level==0:
    cleanZ=z;old=np.load(BASE/'houston'/f'{arm}_{seed}/predictions_200.npz');assert np.array_equal(logits.argmax(1),old['predictions']);assert np.max(abs(logits-old['logits']))<1e-6
   dump(folder/f'{arm}_curve_progress.json',curve);print(seed,arm,'affine',level,'OA',metrics['oa_percent'],'AA',metrics['aa_percent'],flush=True)
  np.save(folder/f'{arm}_target_Z.npy',cleanZ);_,sourceZ,sourceW,sourceT=infer(model,loaders['source'],device,capture=True)
  for key,value in [('source_Z',sourceZ),('source_W',sourceW),('source_T',sourceT)]:np.save(folder/f'{arm}_{key}.npy',value)
  targetW=np.load(BASE/'houston'/f'{arm}_{seed}/weights_200.npy',mmap_mode='r');targetT=np.load(BASE/'houston'/f'{arm}_{seed}/tokens_200.npy',mmap_mode='r')
  spatial={'source':selectivity(sourceW,gs,sourcecoords,shapes['source']),'target':selectivity(targetW,gt,targetcoords,shapes['target'])}
  separation={'T':separability(sourceT,sl,targetT,tl),'Z':separability(sourceZ,sl,cleanZ,tl)}
  assert before==common.state_hash(model);assert cp_hash==common.sha(cp)
  results[arm]={'seed':seed,'checkpoint_sha256':cp_hash,'affine_curve':curve,'spatial_selectivity':spatial,'class_separability':separation,'read_only_state_unchanged':True};dump(folder/f'{arm}_diagnostics.json',results[arm]);del model
 dump(folder/'summary.json',results);print('Completed seed',seed,flush=True)
if __name__=='__main__':main()
