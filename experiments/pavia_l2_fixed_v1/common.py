"""Repair the Pavia cube return only; preserve archived self training protocol."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_l2_fixed_v1')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


base = module('fixed_l2_extension_base', ROOT / 'experiments/mluda_bida_extension_v1/common.py')
DATA, helper, joint, relation, dump = base.DATA, base.helper, base.joint, base.relation, base.dump
historical = base.pavia


def load_scene(directory, name):
    image = loadmat(directory / (name + '.mat'))[name].astype(np.float32)
    labels = loadmat(directory / (name + '_gt_7.mat'))[name + '_gt_7'].astype(int)-1
    assert image.shape[-1]==102 and set(np.unique(labels))==set(range(-1,7))
    original_shape = image.shape
    flat = image.reshape(-1,102)
    index = np.where(flat.sum(axis=-1)!=0)
    values = flat[index]
    values = values / values.max()
    norm = np.sqrt((values**2).sum(1))[:,None].repeat(102,axis=1)
    norm[norm==0]=1
    flat[index] = values/norm
    # Essential repair: reshape may have copied an F-contiguous MATLAB cube.
    image = flat.reshape(original_shape)
    assert np.isfinite(image).all()
    return image, labels


# Facade avoids mutating the imported historical module or its function globals.
pavia = SimpleNamespace(**historical.__dict__)
pavia.load_scene = load_scene


def loaders(task, seed, workers=4):
    assert task=='pavia'
    source, sg = load_scene(DATA / 'Pavia','paviaU')
    target, tg = load_scene(DATA / 'Pavia','pavia')
    train_gt, val_gt = pavia.sample_gt(sg,.95,seed,mode='random')
    target_gt,_ = pavia.sample_gt(tg,1,seed,mode='random')
    pad_image=lambda x:np.pad(x,((6,6),(6,6),(0,0)),mode='reflect')
    pad_gt=lambda x:np.pad(x,((6,6),(6,6)),mode='reflect')
    source,target=pad_image(source),pad_image(target)
    sets=[pavia.HSIDataset(source,pad_gt(train_gt),13,data_aug=True),
          pavia.HSIDataset(source,pad_gt(val_gt),13,data_aug=False),
          pavia.HSIDataset(target,pad_gt(target_gt),13,data_aug=True),
          pavia.HSIDataset(target,pad_gt(target_gt),13,data_aug=False)]
    generator=torch.Generator().manual_seed(seed)
    return [torch.utils.data.DataLoader(base.Unlabeled(ds) if i==2 else ds,
            batch_size=128,shuffle=i in (0,2),generator=generator,num_workers=workers,
            drop_last=False) for i,ds in enumerate(sets)]


def inputs_audit(task, values):
    assert task=='pavia'
    return {'indices_sha256':{str(i):hashlib.sha256(v.dataset.indices.tobytes()).hexdigest()
                             for i,v in enumerate(values)},
            'sample_counts':[len(v.dataset) for v in values],
            'data_sha256':{p.name:helper.file_hash(p) for p in sorted((DATA/'Pavia').glob('*.mat'))},
            'source_split':'95/5 stratified, native random_state=23',
            'normalization':'native spectral L2; repaired flat-to-cube return',
            'target_candidates':'native GT nonzero mask; training labels replaced by zero',
            'patch':13,'bands':102,'classes':7}


def build(task, arm, seed):
    assert task=='pavia'
    model=pavia.BiDAnet(n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3)
    initial=helper.state_hash(model)
    unused=pavia.BiDAnet(n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3)
    del unused
    if arm!='A':joint.install(model,'fixed_mixture')
    if arm=='C':relation(102)[0].install(model,'C',seed)
    return model,initial
