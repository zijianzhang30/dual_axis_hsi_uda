"""Actual final weights/tokens, entropy/diversity and permutation-aware comparisons."""
import argparse
import itertools
import json
import numpy as np
from common import OUT,SEEDS,ARMS,dump
EPS=1e-12

def js(x,y):
 x=np.clip(x,EPS,1);y=np.clip(y,EPS,1);m=(x+y)/2
 return .5*np.sum(x*np.log(x/m)+y*np.log(y/m),axis=-1)
def cosine(x,y):return (x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)+EPS)
def pair(x,y):
 n=len(x);sums=np.zeros(4,dtype=np.float64);perms=list(itertools.permutations(range(4)))
 for start in range(0,n,256):
  a=np.asarray(x[start:start+256],dtype=np.float64);b=np.asarray(y[start:start+256],dtype=np.float64);bs=len(a)
  j=np.empty((bs,4,4));c=np.empty_like(j)
  for i in range(4):
   for k in range(4):j[:,i,k]=js(a[:,i],b[:,k]);c[:,i,k]=cosine(a[:,i],b[:,k])
  costs=np.stack([j[:,range(4),p].mean(1) for p in perms],axis=1);best=costs.argmin(1)
  matched=np.asarray(perms)[best]
  sums[0]+=j[:,range(4),range(4)].mean(1).sum();sums[1]+=c[:,range(4),range(4)].mean(1).sum()
  sums[2]+=j[np.arange(bs)[:,None],np.arange(4),matched].mean(1).sum();sums[3]+=c[np.arange(bs)[:,None],np.arange(4),matched].mean(1).sum()
 return dict(zip(['same_index_JS','same_index_cosine','permutation_aligned_JS','permutation_aligned_cosine'],(sums/n).tolist()))
def diversity(w,t):
 n=len(w);total=np.zeros(7);pairs=list(itertools.combinations(range(4),2))
 for start in range(0,n,256):
  x=np.asarray(w[start:start+256],dtype=np.float64);v=np.asarray(t[start:start+256],dtype=np.float64);bs=len(x)
  entropy=-(x*np.log(np.clip(x,EPS,1))).sum(-1).mean(1)
  wc=np.mean(np.stack([cosine(x[:,i],x[:,j]) for i,j in pairs],1),1)
  wj=np.mean(np.stack([js(x[:,i],x[:,j]) for i,j in pairs],1),1)
  tc=np.mean(np.stack([cosine(v[:,i],v[:,j]) for i,j in pairs],1),1)
  sv=np.linalg.svd(v,compute_uv=False);p=sv/(sv.sum(-1,keepdims=True)+EPS);rank=np.exp(-(p*np.log(np.clip(p,EPS,1))).sum(-1))
  total+=np.array([entropy.sum(),(entropy/np.log(x.shape[-1])).sum(),wc.sum(),wj.sum(),tc.sum(),rank.sum(),np.linalg.norm(v,axis=-1).mean(1).sum()])
 return dict(zip(['weight_entropy_nats','weight_entropy_normalized','weight_pair_cosine','weight_pair_JS','token_pair_cosine','token_effective_rank','token_norm'],(total/n).tolist()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['houston','pavia'],required=True);args=p.parse_args();root=OUT/args.dataset
 assert (root/'all_predictions_frozen.json').exists();report={}
 for seed in SEEDS:
  ws={a:np.load(root/f'{a}_{seed}/weights_200.npy',mmap_mode='r') for a in ARMS};ts={a:np.load(root/f'{a}_{seed}/tokens_200.npy',mmap_mode='r') for a in ARMS}
  variants={a:diversity(ws[a],ts[a]) for a in ARMS};comparisons={f'{a}-{b}':pair(ws[a],ws[b]) for a,b in itertools.combinations(ARMS,2)}
  report[str(seed)]={'variants':variants,'comparisons':comparisons}
  dump(root/f'weight_analysis_{seed}.json',report[str(seed)]);print('Weights analyzed',args.dataset,seed,flush=True)
 aggregate={a:{key:{'mean':float(np.mean([report[str(s)]['variants'][a][key] for s in SEEDS])),'sample_std':float(np.std([report[str(s)]['variants'][a][key] for s in SEEDS],ddof=1))} for key in report[str(SEEDS[0])]['variants'][a]} for a in ARMS}
 dump(root/'weight_analysis.json',{'runs':report,'aggregate':aggregate,'JS_units':'nats','token_effective_rank':'exp entropy of normalized singular values;1-4','permutation_alignment':'per-pixel best of24 token permutations by minimum averageJS; diagnostic only','cross_checkpoint_limit':'A weights use its trained CNN; all comparisons include co-adaptation, not a pure scorer counterfactual.'})
if __name__=='__main__':main()
