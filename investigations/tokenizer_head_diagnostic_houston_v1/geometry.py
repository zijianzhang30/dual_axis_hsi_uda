import numpy as np,json,csv
from common import OUT,CACHE,SEEDS,dump,sha

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def description(x):
 return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'positive_margin_fraction':float((x>0).mean()),'q05':float(np.percentile(x,5)),'q95':float(np.percentile(x,95))}
a=json.loads((OUT/'summary.json').read_text());y=np.load(CACHE/'data/target_labels.npy');rows=[];allseeds={}
for seed in SEEDS:
 z=unit(np.load(CACHE/f'seed_{seed}/target_Z.npy').astype(np.float64));sizes=np.bincount(y,minlength=7);sums=np.stack([z[y==c].sum(0) for c in range(7)]);centers=unit(sums);scores=z@centers.T
 own_LOO=np.einsum('nd,nd->n',z,unit(sums[y]-z));scores[np.arange(len(y)),y]=-np.inf;other=scores.max(1);margin=own_LOO-other
 run=a['runs'][str(seed)];pred=np.load(OUT/f'seed_{seed}/target_fixed_scores.npz')['linear'].argmax(1);source=np.load(OUT/f'seed_{seed}/fixed_head_parameters.npz')['source_centroids'];ws=run['head_parameters']['weight_norm'];w=np.load(OUT/f'seed_{seed}/fixed_head_parameters.npz')['linear_weight'];wc=unit(w);sc=unit(source.astype(np.float64))
 result={}
 for c in range(7):
  which=(y==c);correct=which&(pred==c);wrong=which&(pred!=c)
  d={'all':description(margin[which]),'linear_correct':description(margin[correct]) if correct.any() else None,'linear_wrong':description(margin[wrong]) if wrong.any() else None,'weight_source_centroid_cosine':float(wc[c]@sc[c]),'weight_target_centroid_cosine':float(wc[c]@centers[c]),'source_target_matched_centroid_cosine':float(sc[c]@centers[c])}
  result[str(c+1)]=d
  for subset in ['all','linear_correct','linear_wrong']:
   if d[subset] is not None:rows.append({'seed':seed,'class':c+1,'subset':subset,**d[subset]})
 allseeds[str(seed)]=result
 dump(OUT/f'seed_{seed}/target_class_geometry_offline.json',{'GT_for_class_geometry_only':True,'not_an_inference_head_or_target_fitting':True,'LOO_own_centroid':True,'per_class':result})
with (OUT/'target_class_geometry_offline.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
dump(OUT/'target_class_geometry_offline.json',{'interpretation':'Target GT only defines same-scene class centroids for offline geometry. Positive LOO centroid margin at a linear error demonstrates class structure under GT-conditioned geometry, not source-only separability or a validated replacement head.','seeds':allseeds})
p=json.loads((OUT/'provenance.json').read_text());p['supplementary_geometry_script_sha256']=sha(OUT/'geometry.py');dump(OUT/'provenance.json',p)
for s,r in allseeds.items():print(s,'C6',json.dumps(r['6']))
