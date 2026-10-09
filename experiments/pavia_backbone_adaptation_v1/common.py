"""Matched network-package x official MLUDA objective diagnostic."""
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.dont_write_bytecode=True
ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_backbone_adaptation_v1')
P2=OUT.parent/'pavia_protocol_transfer_v1'
NATIVE=OUT.parent/'mluda_pavia_native_v1/native_snapshot'
SEEDS=(2100,2101,2102)
ARMS=('M_CE','M_AD','O_CE','O_AD')


def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def state_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj);return obj


def base():
    # Own immutable copy of P2's protocol/data-loading utilities and frozen C.
    return module('matrix_protocol',OUT/'snapshot/protocol/common.py')


def official():
    snap=OUT/'snapshot/MLUDA'
    # Avoid MLUDA's flat utils.py shadowing BiDA's namespace utils package.
    sys.modules['Weight']=module('Weight',snap/'Weight.py')
    net=module('matrix_native_net',snap/'net2.py')
    mmd=module('matrix_native_mmd',snap/'mmd.py')
    contrast=module('matrix_native_contrast',snap/'contrastive_loss.py')
    tree=ast.parse((snap/'utils.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('radiation_noise','flip_augmentation')]
    ns={'np':np,'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(snap/'utils.py'),'exec'),ns)
    return net,mmd,contrast,ns


class MNetwork(nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone
    def forward(self,x,t):
        r=self.backbone(x[:,0],t[:,0])
        return {'fs':r[0],'ft':r[5],'ps':r[1],'pt':r[7],'ls':r[3],'lt':r[8]}


class ONetwork(nn.Module):
    def __init__(self,backbone,seed):
        super().__init__();self.backbone=backbone
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(seed+93000).get_state())
            self.head1=nn.Linear(64,128);self.head2=nn.Linear(64,128)
        self._norm_outputs=[]
        backbone.norm.register_forward_hook(self.capture)
    def capture(self,layer,inputs,output):self._norm_outputs.append(output[:,0])
    def forward(self,x,t):
        self._norm_outputs=[];logits=self.backbone(x,t)
        if self.training:
            assert len(self._norm_outputs)==4
            zs,zt=self._norm_outputs[:2]
            return {'fs':zs,'ft':zt,'ps':F.normalize(self.head1(zs),dim=1),
                    'pt':F.normalize(self.head2(zt),dim=1),'ls':logits[0],'lt':logits[1]}
        assert len(self._norm_outputs)==1
        return {'ft':self._norm_outputs[0],'lt':logits[1]}


def build(seed,network):
    b=base();net,_,_,_=official()
    _,worker,_,_=b.environment();worker(seed)
    if network=='M':
        native=net.DSANSS(102,11,7);initial=state_hash(native);model=MNetwork(native)
    else:
        native,_=b.build(seed);initial=state_hash(native)
        before=torch.get_rng_state().clone();model=ONetwork(native,seed)
        assert torch.equal(before,torch.get_rng_state()) and state_hash(native)==initial
    # mmd.py has a global cudnn.benchmark=True side effect; fix both rows to frozen-C deterministic math.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return model,initial


def optimizer(model,network):
    if network=='M':
        b=model.backbone
        groups=[{'params':getattr(b,name).parameters()} for name in ('feature_layers','fc1','fc2','head1','head2')]
    else:groups=[{'params':model.backbone.parameters()},{'params':model.head1.parameters()},{'params':model.head2.parameters()}]
    return torch.optim.SGD(groups,lr=.001,momentum=.9,weight_decay=.0005)


def views(x,t,seed,epoch,step,augmentation):
    # Actual official functions expect B,C,H,W; use that exact layout in both rows.
    rng=np.random.get_state();np.random.seed(seed+epoch*100000+step+60000)
    sx,tx=x[:,0],t[:,0]
    sn=augmentation['radiation_noise'](sx).type(torch.FloatTensor)
    sf=augmentation['flip_augmentation'](sx).type(torch.FloatTensor)
    tn=augmentation['radiation_noise'](tx).type(torch.FloatTensor)
    tf=augmentation['flip_augmentation'](tx).type(torch.FloatTensor)
    np.random.set_state(rng)
    return [(x,t),(sn.unsqueeze(1),tn.unsqueeze(1)),(sf.unsqueeze(1),tf.unsqueeze(1))]


def objective(records,y,epoch,enabled,mmd,contrast):
    ce=F.cross_entropy(records[0]['ls'],y)
    if not enabled:return ce,{'CE':float(ce.detach()),'LMMD':0.,'Con_s':0.,'Con_t':0.,'pseudo_labels_used':False}
    main=records[0]
    pseudo=main['lt'].detach().softmax(1).argmax(1)
    lmmd=mmd.lmmd(main['fs'],main['ft'],y,main['lt'].softmax(1),BATCH_SIZE=32,CLASS_NUM=7)
    cs=contrast(torch.stack((records[1]['ps'],records[2]['ps']),dim=1),y)
    ct=contrast(torch.stack((records[1]['pt'],records[2]['pt']),dim=1),pseudo)
    lambd=2/(1+math.exp(-10*epoch/100))-1
    loss=ce+0.3*lambd*lmmd+ct+cs
    assert torch.isfinite(loss).all()
    return loss,{'CE':float(ce.detach()),'LMMD':float(lmmd.detach()),'Con_s':float(cs.detach()),
                 'Con_t':float(ct.detach()),'lambda':lambd,'pseudo_labels_used':True,
                 'pseudo_counts':torch.bincount(pseudo,minlength=7).detach().cpu().tolist()}


def bn_counts(model):
    return {k:int(v.num_batches_tracked) for k,v in model.named_modules() if isinstance(v,(nn.BatchNorm2d,nn.BatchNorm3d))}


def verify_files():
    m=json.loads((OUT/'provenance.json').read_text())
    for path,h in m['sha256'].items():assert sha(path)==h,path


def trace_hash(arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
