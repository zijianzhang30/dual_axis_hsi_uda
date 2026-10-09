"""Leave-one-objective-out; reuse the immutable paired matrix implementation."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_backbone_adaptation_v1')
OUT=BASE.parent/'pavia_objective_components_v1'
spec=importlib.util.spec_from_file_location('frozen_matrix_common',BASE/'common.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
SEEDS=legacy.SEEDS
ARMS=('CE','Full','Full_no_LMMD','Full_no_source_contrastive','Full_no_target_contrastive')
NEW_ARMS=ARMS[2:]
P2=legacy.P2
dump,sha,digest,state_hash=legacy.dump,legacy.sha,legacy.digest,legacy.state_hash
base,official,build,optimizer,views=legacy.base,legacy.official,legacy.build,legacy.optimizer,legacy.views
bn_counts,trace_hash=legacy.bn_counts,legacy.trace_hash


def run_folder(arm,seed):
    if arm in ('CE','Full'):return BASE/('%s_%d'%({'CE':'O_CE','Full':'O_AD'}[arm],seed))
    return OUT/('%s_%d'%(arm,seed))


def flags(arm):
    return {'LMMD':arm not in ('CE','Full_no_LMMD'),
            'Con_s':arm not in ('CE','Full_no_source_contrastive'),
            'Con_t':arm not in ('CE','Full_no_target_contrastive')}


def objective(records,y,epoch,arm,mmd,contrast):
    if arm in ('CE','Full'):
        return legacy.objective(records,y,epoch,arm=='Full',mmd,contrast)
    assert arm in NEW_ARMS
    ce=F.cross_entropy(records[0]['ls'],y)
    main=records[0];pseudo=main['lt'].detach().softmax(1).argmax(1)
    # Evaluate removed terms for diagnostics only. They are absent from the optimized graph.
    lmmd=mmd.lmmd(main['fs'],main['ft'],y,main['lt'].softmax(1),BATCH_SIZE=32,CLASS_NUM=7)
    cs=contrast(torch.stack((records[1]['ps'],records[2]['ps']),dim=1),y)
    ct=contrast(torch.stack((records[1]['pt'],records[2]['pt']),dim=1),pseudo)
    lambd=2/(1+math.exp(-10*epoch/100))-1
    if arm=='Full_no_LMMD':loss=ce+ct+cs
    elif arm=='Full_no_source_contrastive':loss=ce+.3*lambd*lmmd+ct
    else:loss=ce+.3*lambd*lmmd+cs
    assert torch.isfinite(loss).all()
    return loss,{'CE':float(ce.detach()),'LMMD':float(lmmd.detach()),'Con_s':float(cs.detach()),
                 'Con_t':float(ct.detach()),'lambda':lambd,'enabled':flags(arm),
                 'removed_terms_logged_but_not_optimized':True,'pseudo_labels_used':True,
                 'pseudo_counts':torch.bincount(pseudo,minlength=7).detach().cpu().tolist()}


def verify_files():
    for path,h in json.loads((OUT/'provenance.json').read_text())['sha256'].items():
        assert sha(path)==h,path


def forbid_target_label_loading():
    original=np.load
    def checked(path,*args,**kwargs):
        assert 'evaluation_labels_OFFLINE_ONLY' not in str(path),'Target GT accessed before frozen-endpoint barrier'
        return original(path,*args,**kwargs)
    np.load=checked


def audit_reuse():
    results={}
    for seed in SEEDS:
        for arm in ('CE','Full'):
            folder=run_folder(arm,seed);m=json.loads((folder/'frozen_manifest.json').read_text())
            assert m['epoch']==100 and m['no_target_scoring'] and m['checkpoints'][-1]['epoch']==100
            for item in [m['prediction']]+m['checkpoints']:assert sha(item['path'])==item['sha256']
            assert sha(folder/'last_source_reference.npy')==m['source_reference_sha256']
            rows=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
            assert len(rows)==100 and all(r['epoch']==i+1 and r['steps']==38 and not r['target_scoring'] for i,r in enumerate(rows))
            results['%s_%d'%(arm,seed)]={'folder':str(folder),'manifest_sha256':sha(folder/'frozen_manifest.json'),
                  'history_sha256':sha(folder/'history.jsonl'),'prediction':m['prediction'],
                  'epoch100_checkpoint':m['checkpoints'][-1],'config':m['config']}
        a,b=[json.loads((run_folder(arm,seed)/'frozen_manifest.json').read_text()) for arm in ('CE','Full')]
        for k in ('initial_base_model_sha256','initial_complete_model_sha256','parameters'):assert a['config'][k]==b['config'][k]
    return results
