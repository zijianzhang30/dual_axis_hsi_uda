"""CE+LMMD simplicity and original/relation tokenizer control; immutable matrix code."""
import importlib.util
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_backbone_adaptation_v1')
COMP=BASE.parent/'pavia_objective_components_v1'
OUT=BASE.parent/'pavia_lmmd_tokenizer_v1'
spec=importlib.util.spec_from_file_location('frozen_lmmd_matrix',BASE/'common.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
SEEDS=legacy.SEEDS
ARMS=('C_CE','C_LMMD','C_LMMD_CT','A_LMMD')
NEW_ARMS=('C_LMMD','A_LMMD')
P2=legacy.P2
dump,sha,digest,state_hash=legacy.dump,legacy.sha,legacy.digest,legacy.state_hash
base,official,optimizer,views=legacy.base,legacy.official,legacy.optimizer,legacy.views
bn_counts,trace_hash=legacy.bn_counts,legacy.trace_hash

def run_folder(arm,seed):
    if arm=='C_CE':return BASE/('O_CE_%d'%seed)
    if arm=='C_LMMD_CT':return COMP/('Full_no_source_contrastive_%d'%seed)
    assert arm in NEW_ARMS
    return OUT/('%s_%d'%(arm,seed))

def shared_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith(('backbone.conv_a.','backbone.relation_scores.')):continue
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def build(seed,tokenizer):
    assert tokenizer in ('A','C')
    if tokenizer=='C':return legacy.build(seed,'O')
    b=base();official();BiDAnet,worker,joint,_=b.environment();worker(seed)
    model=BiDAnet(n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3)
    unused=BiDAnet(n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3);del unused
    ref=json.loads((b.P0/'pavia'/('C_%d'%seed)/'config.json').read_text())
    assert b.state_hash(model)==ref['initial_original_model_sha256']
    joint.install(model,'fixed_mixture');initial=state_hash(model)
    before=torch.get_rng_state().clone();wrapped=legacy.ONetwork(model,seed)
    assert torch.equal(before,torch.get_rng_state()) and state_hash(model)==initial
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return wrapped,initial

def flags(arm):return {'LMMD':arm!='C_CE','Con_s':False,'Con_t':arm=='C_LMMD_CT'}

def objective(records,y,epoch,arm,mmd,contrast):
    assert arm in NEW_ARMS
    ce=F.cross_entropy(records[0]['ls'],y);main=records[0]
    soft=main['lt'].softmax(1)
    lmmd=mmd.lmmd(main['fs'],main['ft'],y,soft,BATCH_SIZE=32,CLASS_NUM=7)
    lambd=2/(1+math.exp(-10*epoch/100))-1
    loss=ce+.3*lambd*lmmd;assert torch.isfinite(loss).all()
    return loss,{'CE':float(ce.detach()),'LMMD':float(lmmd.detach()),'Con_s':0.,'Con_t':0.,
        'lambda':lambd,'LMMD_weight':.3*lambd,'pseudo_labels_used':True,
        'pseudo_labels_usage':'native LMMD weights only; no pseudo-label contrastive CE or supervision',
        'pseudo_counts':torch.bincount(soft.detach().argmax(1),minlength=7).cpu().tolist()}

def verify_files():
    for path,h in json.loads((OUT/'provenance.json').read_text())['sha256'].items():assert sha(path)==h,path

def forbid_target_label_loading():
    original=np.load
    def checked(path,*args,**kwargs):
        assert 'evaluation_labels_OFFLINE_ONLY' not in str(path),'Target GT accessed before endpoint barrier'
        return original(path,*args,**kwargs)
    np.load=checked

def audit_reuse():
    result={}
    for seed in SEEDS:
        for arm in ('C_CE','C_LMMD_CT'):
            p=run_folder(arm,seed);m=json.loads((p/'frozen_manifest.json').read_text())
            assert m['epoch']==100 and m['no_target_scoring'] and m['checkpoints'][-1]['epoch']==100
            for v in [m['prediction']]+m['checkpoints']:assert sha(v['path'])==v['sha256']
            assert sha(p/'last_source_reference.npy')==m['source_reference_sha256']
            h=[json.loads(v) for v in (p/'history.jsonl').read_text().splitlines()]
            assert len(h)==100 and all(r['steps']==38 and r['epoch']==i+1 and not r['target_scoring'] for i,r in enumerate(h))
            result['%s_%d'%(arm,seed)]={'folder':str(p),'config':m['config'],
                'manifest_sha256':sha(p/'frozen_manifest.json'),'history_sha256':sha(p/'history.jsonl'),
                'prediction':m['prediction'],'epoch100_checkpoint':m['checkpoints'][-1]}
        aa,bb=[result['%s_%d'%(a,seed)]['config'] for a in ('C_CE','C_LMMD_CT')]
        for k in ('initial_base_model_sha256','initial_complete_model_sha256','parameters'):assert aa[k]==bb[k]
    return result
