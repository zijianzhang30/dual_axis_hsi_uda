"""Frozen existing benchmark protocols, original/self vs relation/self + native LMMD."""
import hashlib
import copy
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/zhangzj26/dual_axis_hsi_uda')
RESULTS=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
OUT=RESULTS/'cross_benchmark_lmmd_v1'
SEEDS=(2100,2101,2102)
PLAN={'datasets':['houston','hyrank'],'arms':['A_LMMD','C_LMMD'],'seeds':list(SEEDS),
 'epochs':200,'patch':13,'batch':128,'optimizer':'persistent SGD lr=.01,momentum0,weight_decay0,no scheduler',
 'BN':'unchanged Full Joint,momentum.19; one paired forward per existing training step; eval running statistics',
 'network':'unchanged original CNN+3 self Transformer blocks; original A vs current nonlinear normalized center-relative C',
 'loss':'source CE + .3*(2/(1+exp(-10*epoch/200))-1)*official MLUDA LMMD',
 'LMMD':'exact existing mmd.py/Weight.py,5 Gaussian kernels,multiplier2,auto detached bandwidth,detached target soft class weights',
 'alignment_features':'actual LayerNorm CLS64 before existing classifier; no new trainable heads or projection',
 'augmentation':'existing benchmark augmentation only; no extra radiation/flip views or BN updates',
 'sampling':'unchanged benchmark95% source train split and native target candidate mask; existing source-val schedule',
 'formal_checkpoint':'fixed epoch200 only; no target-best or target scoring during training',
 'comparisons':['C_LMMD-A_LMMD','C_LMMD-C_CE','A_LMMD-A_CE'],
 'HyRANK_A_definition':'original tokenizer + Full Joint BN + self CE/LMMD; historical B, NOT full released BiDA A',
 'HyRANK_evaluation':'native historical score primary; original-scene clean-coordinate subset separately, never mix',
 'followups':'no automatic parameter sweeps, new benchmarks, variants or losses'}

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def state_hash(model):
    h=hashlib.sha256()
    for value in model.state_dict().values():h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def shared_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith(('conv_a.','relation_scores.')):continue
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def environment(dataset):
    if dataset=='houston':
        m=module('frozen_houston_protocol',RESULTS/'tokenizer_mechanism_v2/common.py')
        sys.path.insert(0,str(RESULTS/'tokenizer_mechanism_v2'))
        env=m.environment()
        return m,env[4],env[6]
    m=module('frozen_hyrank_protocol',ROOT/'experiments/official_bench_v1/common.py')
    return m,m.seed_worker,m.helper

def baseline(dataset,arm,seed):
    if dataset=='houston':return RESULTS/'tokenizer_mechanism_v2/houston'/('%s_%d'%(arm,seed))
    return RESULTS/'official_bench_v1'/('hyrank_%s_%d'%('B' if arm=='A' else 'C',seed))

def setup(dataset,arm,seed):
    m,seed_worker,helper=environment(dataset);seed_worker(seed)
    if dataset=='houston':
        args=Namespace(dataset='houston',seed=seed,num_workers=4,dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/')
        loaders=m.loaders(args);model,initial=m.build(seed,arm,dataset);classes=7
        idx={n:sha_array(l.dataset.indices) for n,l in zip(('source','val','target','target_eval'),loaders)}
        rc=json.loads((baseline(dataset,arm,seed)/'config.json').read_text())
        assert idx==rc['indices_sha256'] and initial==rc['initial_original_model_sha256']
        assert state_hash(model)==rc['initial_full_model_sha256']
    else:
        args=m.options(seed);loaders,audit=m.loaders(args)
        model,ema,initial=m.build(args,'B' if arm=='A' else 'C');assert ema is None;classes=12
        rc=json.loads((baseline(dataset,arm,seed)/'config.json').read_text())
        assert audit==rc['data_protocol'] and initial==rc['initial_common_model_sha256']
        assert state_hash(model)==rc['initial_complete_state_sha256']
        idx={n:sha_array(l.dataset.indices) for n,l in zip(('source','val','target','target_eval'),loaders)}
    return model,loaders,helper,classes,{'indices_sha256':idx,'original_initial_hash':initial,
        'complete_initial_hash':state_hash(model),'shared_initial_hash':shared_hash(model)}

def sha_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def lmmd():
    p=RESULTS/'pavia_backbone_adaptation_v1/snapshot/MLUDA'
    sys.modules['Weight']=module('Weight',p/'Weight.py');m=module('fixed_native_lmmd',p/'mmd.py')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return m

def clean_dataset(native,keep_labels=False):
    """Include original-scene row0/col0 centers, excluded by native strict boundary rule."""
    out=copy.copy(native);radius=6
    x,y=np.nonzero(native.label>=0)
    valid=(x>=radius)&(x<native.data.shape[0]-radius)&(y>=radius)&(y<native.data.shape[1]-radius)
    out.indices=np.column_stack((x[valid],y[valid]))
    out.data_aug=False
    out.labels=[int(native.label[a,b]) for a,b in out.indices] if keep_labels else np.zeros(len(out.indices),dtype=np.int64)
    if not keep_labels:out.label=np.zeros_like(native.label)
    return out

def verify_files():
    for path,h in json.loads((OUT/'provenance.json').read_text()).items():assert sha(path)==h,path
