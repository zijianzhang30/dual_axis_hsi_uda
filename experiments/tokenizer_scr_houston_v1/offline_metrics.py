import numpy as np
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
