"""Z-score intervention with RNG-neutral actual-batch shadow auditing."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from sklearn import preprocessing

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_zscore_v1')
DATA=Path('/nas1/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


base=module('zscore_extension_base',ROOT/'experiments/mluda_bida_extension_v1/common.py')
pavia=base.pavia;helper=base.helper


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def reference(arm):
    return (ROOT/'results/pavia_joint_v1/joint_2100' if arm=='B'
            else ROOT/'results/mluda_bida_extension_v1/pavia/C_2100')


def load_scene(name):
    image=loadmat(DATA/(name+'.mat'))[name]
    gt=loadmat(DATA/(name+'_gt_7.mat'))[name+'_gt_7'].astype(int)-1
    assert image.shape[-1]==102 and set(np.unique(gt))==set(range(-1,7))
    flat=preprocessing.scale(image.reshape(-1,102))
    z=flat.reshape(image.shape)
    assert np.isfinite(z).all()
    return z,gt


class IndexedDataset(pavia.HSIDataset):
    def __getitem__(self,index):
        x,y=super().__getitem__(index)
        return x,y,index


class Trace:
    def __init__(self):self.source=None;self.reset()
    def reset(self):
        self.shadow=hashlib.sha256();self.indices=hashlib.sha256();self.steps=0
    def paired(self,x,y,indices):
        source,labels,si=self.source
        if len(source)!=len(x):return
        for t in (source[:,:,102:].contiguous(),labels,x[:,:,102:].contiguous()):
            self.shadow.update(t.numpy().tobytes())
        self.indices.update(si.numpy().tobytes());self.indices.update(indices.numpy().tobytes())
        self.steps+=1


class Loader:
    def __init__(self,loader,trace,kind):self.loader,self.trace,self.kind=loader,trace,kind
    def __getattr__(self,key):return getattr(self.loader,key)
    def __iter__(self):
        for full,labels,indices in self.loader:
            assert full.shape[2]==204
            if self.kind=='source':self.trace.source=(full,labels,indices)
            if self.kind=='target':self.trace.paired(full,labels,indices)
            yield full[:,:,:102].contiguous(),torch.zeros_like(labels) if self.kind=='target' else labels


def loaders(seed=2100,workers=4,trace=None):
    trace=trace or Trace()
    joined=[];gts=[]
    for name in ('paviaU','pavia'):
        z,gt=load_scene(name)
        shadow,old_gt=pavia.load_scene(DATA,name)
        assert np.array_equal(gt,old_gt)
        # Spatial augmentation acts identically on all 204 channels; only the
        # first 102 standardized channels are ever passed to model.forward.
        joined.append(np.concatenate((z,shadow),axis=-1));gts.append(gt)
    train_gt,val_gt=pavia.sample_gt(gts[0],.95,seed,mode='random')
    target_gt,_=pavia.sample_gt(gts[1],1,seed,mode='random')
    pi=lambda a:np.pad(a,((6,6),(6,6),(0,0)),mode='reflect')
    pg=lambda a:np.pad(a,((6,6),(6,6)),mode='reflect')
    source,target=map(pi,joined)
    datasets=[IndexedDataset(source,pg(train_gt),13,data_aug=True),
              IndexedDataset(source,pg(val_gt),13,data_aug=False),
              IndexedDataset(target,pg(target_gt),13,data_aug=True),
              IndexedDataset(target,pg(target_gt),13,data_aug=False)]
    generator=torch.Generator().manual_seed(seed)
    raw=[torch.utils.data.DataLoader(ds,batch_size=128,num_workers=workers,
         shuffle=i in (0,2),drop_last=False,generator=generator) for i,ds in enumerate(datasets)]
    return [Loader(x,trace,kind) for x,kind in zip(raw,('source','val','target','eval'))]


def audit(values):
    return {'indices_sha256':{str(i):hashlib.sha256(x.dataset.indices.tobytes()).hexdigest()
                             for i,x in enumerate(values)},
            'sample_counts':dict(zip(('source_train','source_val','target_train','target_eval'),
                                    (len(x.dataset) for x in values))),
            'data_sha256':{p.name:helper.file_hash(p) for p in sorted(DATA.glob('*.mat'))}}


def official_scene(name):
    path=Path('/home/zhangzj26/TGRS_MLUDA-2024-main/utils.py')
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_data_pavia')
    ns={'sio':__import__('scipy.io',fromlist=['loadmat']),'np':np,'preprocessing':preprocessing}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    return ns['load_data_pavia'](str(DATA/(name+'.mat')),str(DATA/(name+'_gt_7.mat')))
