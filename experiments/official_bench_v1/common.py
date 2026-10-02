"""Released loaders/models/losses, plus a band-only relation adapter."""
import ast
import argparse
import difflib
import hashlib
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
BIDA=Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
OUT=ROOT/'results/official_bench_v1'
DATA=Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')
sys.path.insert(0,str(BIDA))
from models.get_model import get_model
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils_HSI import seed_worker
from utils.scheduler import load_scheduler
from loss import make_loss
import train_pipeline as official


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


helper=module('official_metrics',ROOT/'experiments/bida_self_dropout/train.py')
joint=module('official_joint',ROOT/'experiments/bn_stabilization_v1/normalization.py')


def relation_adapter(bands):
    path=ROOT/'experiments/relation_tokenizer_v1/relation.py'
    original=path.read_text()
    derived=original
    for old,new in [('bands=48',f'bands={bands}'),('(1, 48)',f'(1, {bands})'),
                    ('B×1×48×H×W',f'B×1×{bands}×H×W')]:
        assert derived.count(old)==1;derived=derived.replace(old,new)
    value=types.ModuleType(f'official_relation_{bands}')
    exec(compile(derived,str(path), 'exec'),value.__dict__)
    return value,derived,''.join(difflib.unified_diff(original.splitlines(True),derived.splitlines(True),
                        fromfile=str(path),tofile=f'relation_{bands}_bands.py'))


def native_train():
    """Only remove target metrics/checkpoint selection, retain native optimization."""
    path=BIDA/'train_pipeline.py'
    tree=ast.parse(path.read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='train')
    fn.args.args.append(ast.arg(arg='frozen_callback'))
    loops=[x for x in fn.body if isinstance(x,ast.For)]
    assert len(loops)==1
    loop=loops[0]
    blocks=[x for x in loop.body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare)
            and isinstance(x.test.left,ast.BinOp) and isinstance(x.test.left.op,ast.Mod)
            and isinstance(x.test.left.right,ast.Attribute) and x.test.left.right.attr=='log_interval']
    assert len(blocks)==1
    removed=blocks[0]
    call=ast.parse('frozen_callback(e, network, network_ema, optimizer, global_step)').body[0]
    loop.body[loop.body.index(removed)]=call
    ast.fix_missing_locations(fn)
    namespace=dict(official.__dict__)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    # Record the exact removed block and replacement; native inner loop untouched.
    lines=path.read_text().splitlines(True)
    old=''.join(lines[removed.lineno-1:removed.end_lineno])
    diff=''.join(difflib.unified_diff(old.splitlines(True),['        frozen_callback(e, network, network_ema, optimizer, global_step)\n'],
                                   fromfile='released_target_selection_block',tofile='target_blind_callback'))
    return namespace['train'],diff


def options(seed=2100):
    return argparse.Namespace(model='BiDA',source_name='Dioni',target_name='Loukia',
             dataset_dir=str(DATA)+'/',patch_size=13,epoch=200,epochs=200,bs=128,lr=.01,
             ratio=.95,ema_decay=.999,dim=64,depth=3,num_tokens=4,num_workers=4,
             loss_type='softmax',labelsmooth='off',lambda1=.1,lambda2=1.,log_interval=10,re_ratio=1,seed=seed)


def loaders(args):
    source,sgt,source_classes=load_mat_hsi('Dioni',args.dataset_dir,norm='normband')
    target,tgt,target_classes=load_mat_hsi('Loukia',args.dataset_dir,norm='normband')
    assert source.shape==(250,1376,176) and target.shape==(249,945,176)
    assert source_classes==target_classes and len(source_classes)==12
    assert set(np.unique(sgt))==set(range(-1,12)) and set(np.unique(tgt))==set(range(-1,12))
    assert np.isfinite(source).all() and np.isfinite(target).all()
    raw={'source_mask_centers':int((sgt>=0).sum()),'target_mask_centers':int((tgt>=0).sum())}
    train_gt,val_gt=sample_gt(sgt,args.ratio,args.seed,mode='random')
    target_gt,_=sample_gt(tgt,1,args.seed,mode='random')
    image_pad=lambda a:np.pad(a,((6,6),(6,6),(0,0)),mode='reflect')
    gt_pad=lambda a:np.pad(a,((6,6),(6,6)),mode='reflect')
    source,target=image_pad(source),image_pad(target)
    datasets=[HSIDataset(source,gt_pad(train_gt),13,data_aug=True),
              HSIDataset(source,gt_pad(val_gt),13,data_aug=False),
              HSIDataset(target,gt_pad(target_gt),13,data_aug=True),
              HSIDataset(target,gt_pad(target_gt),13,data_aug=False)]
    g=torch.Generator().manual_seed(args.seed)
    values=[torch.utils.data.DataLoader(d,batch_size=128,num_workers=args.num_workers,
               shuffle=i in (0,2),drop_last=False,generator=g) for i,d in enumerate(datasets)]
    audit={'bands':176,'classes':source_classes,'class_count':12,**raw,
           'counts':dict(zip(('source_train','source_val','target_train','target_eval'),map(len,datasets))),
           'index_sha256':{str(i):hashlib.sha256(d.indices.tobytes()).hexdigest() for i,d in enumerate(datasets)},
           'target_candidate_mask':'author out68 GT !=0; values ignored in training',
           'eval_mask':'same nonzero mask, native HSIDataset x>radius,y>radius boundary rule',
           'data_sha256':{name:helper.file_hash(DATA/name) for name in
                         ('Dioni.mat','Loukia.mat','Dioni_gt_out68.mat','Loukia_gt_out68.mat')}}
    return values,audit


def build(args,arm):
    model=get_model('BiDA','Dioni',13,args)
    initial=helper.state_hash(model)
    ema=get_model('BiDA','Dioni',13,args,ema=True)
    if arm!='A':
        del ema;ema=None;joint.install(model,'fixed_mixture')
    if arm=='C':
        adapter,_,_=relation_adapter(176);adapter.install(model,'C',args.seed)
    return model,ema,initial
