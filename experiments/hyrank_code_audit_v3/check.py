"""Read-only code/data audit plus isolated CPU update equivalence (not training)."""
import ast
import builtins
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import re
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_code_audit_v3')
PINNED=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_repro_audit_v2/dependency_snapshot/official_BiDA_pinned')
sys.path.insert(0,str(ROOT/'experiments/official_bench_v1'))
import common as c


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def sha(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def rng():return (torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),copy.deepcopy(random.getstate()))


def restore(state):
    torch.set_rng_state(state[0]);np.random.set_state(state[1]);random.setstate(state[2])


def rng_hash(state):return hashlib.sha256(pickle.dumps((state[0].numpy(),state[1],state[2]))).hexdigest()


def compare_tensors(left,right):
    assert left.keys()==right.keys()
    worst={'max_abs_error':0.,'first_nonzero_key':None,'worst_key':None}
    for key in left:
        a,b=left[key],right[key]
        if a is None or b is None:
            assert a is None and b is None,key
            continue
        assert a.shape==b.shape,key
        error=float((a.double()-b.double()).abs().max())
        assert np.isfinite(error),key
        if error>0 and worst['first_nonzero_key'] is None:worst['first_nonzero_key']=key
        if error>worst['max_abs_error']:worst.update(max_abs_error=error,worst_key=key)
    return worst


def body(fn):
    outer=next(x for x in fn.body if isinstance(x,ast.For))
    return next(x for x in outer.body if isinstance(x,ast.For)).body


def compile_step(nodes,namespace):
    args='network, network_ema, optimizer, criterion, MMD_criterion, data_src, data_tar, opts, device, e, global_step'
    skeleton=ast.parse('def diagnostic_step('+args+'):\n    pass\n').body[0]
    skeleton.body=copy.deepcopy(nodes)+ast.parse('''
return {'logits': {'source':out_x.detach().clone(),'target':out_x_tar.detach().clone(),
                  'fusion':out_x_fusion.detach().clone(),'fusion_source':out_x_fusion_src.detach().clone()},
        'losses': {'ce':loss_cls.detach().clone(),'distill_target':loss_dis.detach().clone(),
                   'distill_source':loss_dis_src.detach().clone(),'consistency_target':loss_con_tar.detach().clone(),
                   'consistency_source':loss_con_src.detach().clone(),'total':loss.detach().clone(),
                   'mmd':Align_loss.detach().clone() if e>100 else loss.new_zeros(())},
        'global_step':global_step,'mmd_active':e>100}
''').body
    ast.fix_missing_locations(skeleton)
    namespace=dict(namespace)
    namespace.update(losses=[],Align_losses=[],Distill_losses=[])
    exec(compile(ast.Module(body=[skeleton],type_ignores=[]),'isolated_original_batch_AST','exec'),namespace)
    return namespace['diagnostic_step']


def run_case(name,schedule,student,teacher,optimizer_state,initial_rng,batches,opts,functions):
    outputs=[]
    for function in functions:
        model=copy.deepcopy(student);ema=copy.deepcopy(teacher)
        optimizer=torch.optim.SGD(model.parameters(),lr=opts.lr)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        assert {n:m.training for n,m in model.named_modules()}=={n:m.training for n,m in student.named_modules()}
        restore(initial_rng); trace=[]
        for i,(epoch,step) in enumerate(schedule):
            model.train()
            result=function(model,ema,optimizer,F.cross_entropy,c.official.MMD_loss(),
                            batches[0],batches[1],opts,torch.device('cpu'),epoch,step)
            trace.append({'logits':result['logits'],'losses':result['losses'],
                'global_step':result['global_step'],'mmd_active':result['mmd_active'],
                'gradients':{n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()},
                'student_parameters':{n:p.detach().clone() for n,p in model.named_parameters()},
                'student_buffers':{n:p.detach().clone() for n,p in model.named_buffers()},
                'ema_parameters':{n:p.detach().clone() for n,p in ema.named_parameters()},
                'ema_buffers':{n:p.detach().clone() for n,p in ema.named_buffers()},
                'student_modes':{n:m.training for n,m in model.named_modules()},
                'ema_modes':{n:m.training for n,m in ema.named_modules()},
                'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':rng_hash(rng())})
        outputs.append(trace)
    comparisons=[]
    for i,(left,right) in enumerate(zip(*outputs)):
        metrics={key:compare_tensors(left[key],right[key]) for key in
                 ('logits','losses','gradients','student_parameters','student_buffers','ema_parameters','ema_buffers')}
        assert all(left[k]==right[k] for k in ('global_step','mmd_active','student_modes','ema_modes','rng','optimizer'))
        comparisons.append({'epoch':schedule[i][0],'global_step':left['global_step'],
                            'mmd_active':left['mmd_active'],'errors':metrics,
                            'native_losses':{k:float(v) for k,v in left['losses'].items()},
                            'modes_optimizer_rng_equal':True})
        assert all(v['max_abs_error']==0 for v in metrics.values()),'First numerical divergence: '+name
    return {'name':name,'batch_size_source':128,'batch_size_target':128,'steps':comparisons,'passed':True}


def main():
    assert not OUT.exists(),'Do not overwrite audit'
    OUT.mkdir(parents=True); (OUT/'check.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2); start=time.monotonic()
    comparisons={}
    files=('main.py','train_pipeline.py','models/BiDA.py','models/get_model.py','utils/dataset.py',
           'utils/scheduler.py','utils/utils_HSI.py','loss/make_loss.py','loss/mmd_loss.py','loss/center_loss.py')
    saved=json.loads((Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_native_entry_v1')/'source_hashes.json').read_text())
    for file in files:
        remote=(PINNED/file).read_text();local=(c.BIDA/file).read_text()
        comparisons[file]={'AST_equal':ast.dump(ast.parse(remote))==ast.dump(ast.parse(local)),
                          'matches_actual_native_run_hash':c.helper.file_hash(c.BIDA/file)==saved[file],
                          'pinned_sha256':c.helper.file_hash(PINNED/file),'local_sha256':c.helper.file_hash(c.BIDA/file)}
        if not comparisons[file]['AST_equal']:
            (OUT/(file.replace('/','_')+'.patch')).write_text(''.join(difflib.unified_diff(remote.splitlines(True),local.splitlines(True))))
    assert all(v['AST_equal'] for k,v in comparisons.items() if k!='utils/dataset.py')
    assert all(v['matches_actual_native_run_hash'] for v in comparisons.values())
    dump('code_comparison.json',comparisons)
    # Independent pinned loader (one necessary constructor assignment in memory).
    raw=(PINNED/'utils/dataset.py').read_text()
    assert raw.count('        self.label = gt')==1
    raw=raw.replace('        self.label = gt','        self.data = image\n        self.label = gt')
    reference=types.ModuleType('isolated_pinned_dataset')
    exec(compile(raw,'pinned_dataset_single_runtime_repair','exec'),reference.__dict__)
    data={}; paper={'Dioni':[1262,204,614,150,1768,361,5035,6374,1754,492,1612,398],
                    'Loukia':[206,54,426,79,1107,422,2996,2361,399,453,1393,421]}
    for scene in paper:
        x,y,classes=c.load_mat_hsi(scene,str(c.DATA)+'/',norm='normband')
        rx,ry,rc=reference.load_mat_hsi(scene,str(c.DATA)+'/',norm='normband')
        counts=np.bincount(y[y>=0],minlength=12).tolist()
        assert np.array_equal(x,rx) and np.array_equal(y,ry) and classes==rc and counts==paper[scene]
        data[scene]={'shape':list(x.shape),'normalization_max_error':float(np.abs(x-rx).max()),
                     'classes':classes,'raw_class_counts':counts,'paper_Table_V_counts_equal':True,
                     'pixel_l2_norm_range':[float(np.linalg.norm(x,axis=-1).min()),float(np.linalg.norm(x,axis=-1).max())]}
        del x,rx,y,ry
    c.seed_worker(2100);loaders,protocol=c.loaders(c.options())
    ds,td=loaders[0].dataset,loaders[2].dataset
    for i,loader in enumerate(loaders):
        native=loader.dataset
        rds=reference.HSIDataset(native.data,native.label,13,data_aug=False)
        assert np.array_equal(native.indices,rds.indices)
        native.data_aug=False
        assert all(torch.equal(native[k][0],rds[k][0]) for k in (0,len(native)//2,len(native)-1))
    dump('data_checks.json',{'scenes':data,'protocol':protocol,'indices_and_sample_patches_equal':True,
         'constructor_runtime_repair':'self.data=image only, in pinned in-memory reference',
         'boundary_score_only_max_difference_pp':30/10317*100})
    # Compare actual compiled local A_no_extra_val batch AST with pinned batch AST.
    control=c.module('audit_control_v3',ROOT/'experiments/hyrank_A_no_extra_val_v1/control.py')
    captured={}
    def capture(tree,*args,**kwargs):
        captured['fn']=copy.deepcopy(tree.body[0]);return builtins.compile(tree,*args,**kwargs)
    control.compile=capture
    control.native(types.SimpleNamespace(out=OUT))
    official_fn=next(n for n in ast.parse((PINNED/'train_pipeline.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    assert ast.dump(ast.Module(body=body(official_fn),type_ignores=[]))==ast.dump(ast.Module(body=body(captured['fn']),type_ignores=[]))
    pinned_namespace={}
    exec(compile((PINNED/'train_pipeline.py').read_text(),'pinned_train_pipeline','exec'),pinned_namespace)
    functions=[compile_step(body(official_fn),pinned_namespace),compile_step(body(captured['fn']),c.official.__dict__)]
    selection=np.random.RandomState(7331)
    si=selection.choice(len(ds),128,replace=False);ti=selection.choice(len(td),128,replace=False)
    source=torch.stack([ds[int(k)][0] for k in si]);labels=torch.as_tensor([ds.labels[int(k)] for k in si])
    target=torch.stack([td[int(k)][0] for k in ti]);unlabeled=torch.zeros(128,dtype=torch.long)
    assert source.shape==target.shape==(128,1,176,13,13)
    assert torch.isfinite(source).all() and torch.isfinite(target).all()
    opts=c.options(2100);c.seed_worker(2100)
    student,teacher,initial=c.build(opts,'A')
    optimizer,scheduler=c.load_scheduler('BiDA',student,opts);assert scheduler is None
    # Native make_loss creates an unused random CenterLoss. Reproduce that CPU
    # RNG consumption without allocating CUDA: active labelsmooth-off CE is F.CE.
    from loss.center_loss import CenterLoss
    unused=CenterLoss(num_classes=12,feat_dim=2048,use_gpu=False);del unused
    initial_rng=rng();batches=((source,labels),(target,unlabeled))
    cases=[]
    for name,schedule in [('epoch100_MMD_off',[(100,7920)]),('epoch101_MMD_on',[(101,8000)]),
                          ('six_step_replay_last3_epoch100_first3_epoch101',[(100,7997+i) for i in range(3)]+[(101,8000+i) for i in range(3)])]:
        case=run_case(name,schedule,student,teacher,optimizer.state_dict(),initial_rng,batches,opts,functions)
        cases.append(case);dump('numerical_partial.json',cases);print(name+' passed',flush=True)
    dump('numerical_equivalence.json',{'passed':True,'scope':'isolated CPU FP32; not complete training or GPU trajectory certification',
         'batch_body_AST_identical':True,'initial_student_sha256':initial,
         'source_center_sha256':sha(ds.indices[si]),'target_center_sha256':sha(td.indices[ti]),
         'source_tensor_sha256':sha(source.numpy()),'target_tensor_sha256':sha(target.numpy()),
         'cases':cases,'max_abs_error':0.,'first_divergence':None,
         'limitations':['Fresh initialized student/EMA, not trained epoch100 checkpoint.',
                       'Extracted original batch body; full loader/validation replay not included.',
                       'CPU FP32 only; CUDA RNG not executed or certified.',
                       'Inactive CenterLoss constructed on CPU; active loss is unchanged F.cross_entropy.']})
    metrics={}
    oldsummary=json.loads((Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_A_no_extra_val_v1')/'summary.json').read_text())
    native_root=Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_native_entry_v1')
    for seed in (2100,2101,2102):
        text=(native_root/('seed_%d.log'%seed)).read_text()
        values=[float(v) for v in re.findall(r'\bOA:\s*([0-9.]+)',text)]
        assert len(values)==20
        history=[json.loads(line) for line in (Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/hyrank_A_no_extra_val_v1')/('seed_%d'%seed)/'history.jsonl').read_text().splitlines()]
        local_values=[row['target']['oa_percent'] for row in history if 'target' in row]
        error=max(abs(a-b) for a,b in zip(values,local_values));assert error<1e-10
        metrics[str(seed)]={'native_fixed200':values[-1],'native_oracle':max(values),
                           'all20_OA_max_native_vs_local_error':error}
    dump('native_vs_local_actual_trajectory.json',{'per_seed':metrics,'fixed200':oldsummary['fixed200'],'oracle':oldsummary['oracle']})
    dump('completed.json',{'passed':True,'seconds':time.monotonic()-start,'complete_training_tasks':0,
                          'historical_checkpoints_modified':False,'official_commit':'66a92b542ffa1df3c8b628af3d8bd001b799f1f6'})
    print('Audit complete; no training task added.',flush=True)


if __name__=='__main__':main()
