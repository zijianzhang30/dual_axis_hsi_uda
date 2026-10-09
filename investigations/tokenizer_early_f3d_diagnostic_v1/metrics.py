import numpy as np
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
