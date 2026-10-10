import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/houston_source_supervised_router_v1'
BASE=ROOT/'results/tokenizer_mechanism_v2'
DATA=ROOT/'results/tokenizer_stem_value_diagnostic_v1/data'
GEOM=ROOT/'results/houston_class_geometry_diagnostic_v1'
CACHE=GEOM
ROUTERS=ROOT/'results/houston_routing_signal_diagnostic_v1'
RELIABILITY=ROOT/'results/houston_router_reliability_diagnostic_v1'
SEEDS=(2100,2101,2102)
STAGES=('V','T','Z')
KS=(5,15)
NAMES=('R0','R1','R2','R3','R4','R5','R6')
STAGE={'R0':'Z','R1':'Z','R2':'Z','R3':'T','R4':'T','R5':'V','R6':'V'}
AUGS=[{'name':'original','scale':1.,'gain_sin':0.,'gain_cos2':0.,'offset_cos':0.,'offset_sin':0.,'noise':0.},
 {'name':'affine_low','scale':.9,'gain_sin':.04,'gain_cos2':0.,'offset_cos':.015,'offset_sin':0.,'noise':.005},
 {'name':'affine_high','scale':1.1,'gain_sin':-.04,'gain_cos2':0.,'offset_cos':-.015,'offset_sin':0.,'noise':.005},
 {'name':'spectral_mild','scale':1.,'gain_sin':0.,'gain_cos2':.03,'offset_cos':0.,'offset_sin':.01,'noise':.01}]
CONFIG={'hidden_dim':16,'epochs':60,'batch_size':256,'lr':.001,'weight_decay':.0001,
        'gate_CE':'unweighted','alternative_loss':'partial-label CE over correct alternatives',
        'source_reference_exclude_chebyshev_le':12,'k':list(KS),'augmentation':AUGS,
        'gate_rule':'argmax keep/override, ties keep','alternative_rule':'argmax R1-R6',
        'G1_rule':'R1-R6 unique plurality, highest-vote tie fallback R0',
        'G2_rule':'max source-training native-margin ECDF percentile, ties R0 then router index',
        'Go_rule':'G3 final-classifier OA > original C OA for every seed at BOTH fixed k values; else No-Go',
        'no_checkpoint_or_threshold_selection':True}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def csvwrite(name,rows):
    with (OUT/name).open('x',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(rows)

def confidence(logits,centroid_scores,counts):
    pred=np.empty((len(logits),7),np.int64);native=np.empty((len(logits),7),float)
    top=np.sort(logits,axis=1);margin=top[:,-1]-top[:,-2]
    sh=logits-logits.max(1,keepdims=True);lp=sh-np.log(np.exp(sh).sum(1,keepdims=True))
    arrays=[margin,-(np.exp(lp)*lp).sum(1)];names=['R0_logit_margin','R0_entropy']
    pred[:,0]=logits.argmax(1);native[:,0]=margin
    for idx,stage in [(1,'Z'),(3,'T'),(5,'V')]:
        score=centroid_scores[stage];top=np.sort(score,axis=1)
        pred[:,idx]=score.argmax(1);m=top[:,-1]-top[:,-2];native[:,idx]=m
        arrays.append(m);names.append(f'R{idx}_cosine_margin')
    for idx,stage in [(2,'Z'),(4,'T'),(6,'V')]:
        c=counts[stage];k=c.sum(1);assert (k==k[0]).all();top=np.sort(c,axis=1)
        pred[:,idx]=c.argmax(1);m=(top[:,-1]-top[:,-2])/k;native[:,idx]=m
        arrays.extend([m,top[:,-1]/k]);names.extend([f'R{idx}_vote_margin',f'R{idx}_neighbor_agreement'])
    agree=pred[:,:,None]==pred[:,None,:]
    for i in range(7):
        for j in range(i+1,7):arrays.append(agree[:,i,j].astype(float));names.append(f'R{i}_R{j}_agreement')
    for i,r in enumerate(NAMES):
        other=[s for s in STAGES if s!=STAGE[r]]
        arrays.append(np.mean([agree[:,i,[j for j,q in enumerate(NAMES) if STAGE[q]==s]].mean(1) for s in other],axis=0));names.append(r+'_other_stage_agreement')
    for label,idx in [('centroid',[1,3,5]),('knn',[2,4,6])]:
        arrays.append((pred[:,idx]==pred[:,idx[:1]]).all(1).astype(float));names.append(label+'_all_stage_agreement')
    x=np.stack(arrays,axis=1);assert x.shape[1]==41 and np.isfinite(x).all()
    return x,pred,native,names

class TinySelector(torch.nn.Module):
    def __init__(self):
        super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(41,16),torch.nn.ReLU())
        self.gate=torch.nn.Linear(16,2);self.alternative=torch.nn.Linear(16,6)
    def forward(self,x):
        h=self.body(x);return self.gate(h),self.alternative(h)

def decisions(model,x,pred):
    with torch.inference_mode():g,a=model(torch.tensor(x,dtype=torch.float32))
    g=g.numpy();a=a.numpy();override=g.argmax(1)==1;selected=a.argmax(1)+1
    router=np.where(override,selected,0);route=pred[np.arange(len(pred)),router]
    return route,router,g,a

def majority(pred):
    count=np.stack([(pred[:,1:]==c).sum(1) for c in range(7)],axis=1)
    tied=(count==count.max(1,keepdims=True)).sum(1)>1
    route=count.argmax(1);route[tied]=pred[tied,0];return route,tied

def ecdf_scores(native,reference):
    return np.stack([(np.searchsorted(np.sort(reference[:,j]),native[:,j],side='left')+
        np.searchsorted(np.sort(reference[:,j]),native[:,j],side='right'))/(2*len(reference)) for j in range(7)],axis=1)
