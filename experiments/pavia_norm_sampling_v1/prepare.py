"""Freeze twelve configs first, then run CPU implementation checks."""
import ast
import json
import shutil
import sys
import time
import importlib.metadata
import numpy as np
import torch
from scipy.io import loadmat
from sklearn import preprocessing
import scipy.io as sio
import common as c


def main():
    torch.set_num_threads(2)
    c.OUT.mkdir(parents=True, exist_ok=True)
    assert not (c.OUT/'frozen_config.json').exists(), 'Do not overwrite frozen protocol'
    link = c.ROOT/'results'/c.OUT.name
    if not link.exists():
        link.symlink_to(c.OUT, target_is_directory=True)
    files = list(__import__('pathlib').Path(__file__).parent.glob('*.py')) + [__import__('pathlib').Path(__file__).with_name('PROTOCOL.md')]
    snapshot = c.OUT/'source_snapshot'
    snapshot.mkdir()
    for file in files:
        shutil.copy2(file,snapshot/file.name)
    dependencies = [__import__('pathlib').Path(c.base.__file__),
                    c.ROOT/'experiments/bn_stabilization_v1/normalization.py',
                    __import__('pathlib').Path('/home/zhangzj26/IEEE_TCSVT_BiDA/models/BiDA.py'),
                    __import__('pathlib').Path('/home/zhangzj26/IEEE_TCSVT_BiDA/utils/dataset.py')]
    for i,file in enumerate(dependencies):
        shutil.copy2(file,snapshot/('%d_'%i+file.name))
    common = {'study':'Pavia normalization x source sampling 2x2', 'seeds':[2100,2101,2102],
              'epochs':200,'batch':128,'steps_per_epoch':290,'samples_per_epoch':c.N,
              'lr':.01,'optimizer':'SGD; momentum=0; weight_decay=0; no scheduler',
              'model':'BiDA-self + Full Joint BN momentum .19 + ORIGINAL tokenizer',
              'patch':13,'depth':3,'dropout':.1,'loss':'source CE only',
              'source_split':'native .95 stratified random_state=23; full pool',
              'target_mask':'GT nonzero candidate mask; class values excluded from training',
              'formal_endpoint':200,'diagnostic_grid':list(range(10,201,10)),
              'source_files_sha256':{str(p):c.helper.file_hash(p) for p in files+dependencies},
              'data_sha256':{p.name:c.helper.file_hash(p) for p in (c.base.DATA/'Pavia').glob('*.mat')},
              'stream_offsets':{'uniform':10000,'balanced':20000,'target':30000,'source_aug':40000,'target_aug':50000},
              'epoch_seed_increment':100000,
              'environment':{'python':sys.version,'torch':torch.__version__,
                  **{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','einops')}}}
    c.dump(c.OUT/'frozen_config.json',common)
    for seed in common['seeds']:
        for arm,(norm,policy) in c.ARMS.items():
            run=c.OUT/('%s_%d'%(arm,seed)); run.mkdir()
            c.dump(run/'config.json',dict(common,seed=seed,arm=arm,normalization=norm,sampling=policy))
    start=time.monotonic()
    checks={}
    # Numerical reference is the actual author base-loader function, not ILDA.
    mluda=__import__('pathlib').Path('/home/zhangzj26/TGRS_MLUDA-2024-main/utils.py')
    tree=ast.parse(mluda.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_data_pavia')
    ns={'np':np,'sio':sio,'preprocessing':preprocessing}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(mluda),'exec'),ns)
    shutil.copy2(mluda,snapshot/'MLUDA_reference_utils.py')
    for name in ('paviaU','pavia'):
        actual,gt=c.scene(name,'zscore')
        expected,_=ns['load_data_pavia'](str(c.base.DATA/'Pavia'/(name+'.mat')),str(c.base.DATA/'Pavia'/(name+'_gt_7.mat')))
        err=float(np.max(np.abs(actual-expected)))
        assert err == 0
        l2,_=c.scene(name,'l2')
        raw=loadmat(c.base.DATA/'Pavia'/(name+'.mat'))[name]
        flat=l2.reshape(-1,102); valid=raw.reshape(-1,102).sum(1)!=0
        normerr=float(np.max(np.abs(np.linalg.norm(flat[valid],axis=1)-1)))
        assert normerr<2e-6 and not np.array_equal(l2,raw)
        checks[name]={'zscore_reference_max_error':err,'zscore_return_dtype':str(actual.dtype),
                      'zscore_range':[float(actual.min()),float(actual.max())],
                      'l2_unit_norm_max_error':normerr,'l2_range':[float(l2.min()),float(l2.max())]}
        del actual,expected,l2,raw
    sets=c.datasets(2100,'l2')
    source_labels=np.asarray(sets[0].labels,dtype=np.int64)
    assert np.bincount(source_labels,minlength=7).tolist()==[2901,6261,3498,1263,900,17635,4778]
    index_hashes=[c.digest(x.indices) for x in sets]
    zsets=c.datasets(2100,'zscore')
    assert index_hashes==[c.digest(x.indices) for x in zsets]
    checks['pool']={'counts':[len(x) for x in sets], 'source_class_counts':np.bincount(source_labels,minlength=7).tolist(),
                    'center_hashes':index_hashes,'normalization_pair_centers_equal':True}
    c.dump(c.OUT/'pool.json',checks['pool'])
    # Audit all epoch plans, without iterating any loader.
    plans={}
    for seed in (2100,2101,2102):
        for policy in ('uniform','balanced'):
            total=np.zeros(7,dtype=np.int64); h=__import__('hashlib').sha256()
            for epoch in range(1,201):
                s,t,a,b=c.plan(seed,epoch,policy,source_labels,len(sets[2]))
                assert len(s)==len(t)==c.N and len(np.unique(t))==c.N
                total+=np.bincount(source_labels[s],minlength=7)
                for array in (s,t,a,b): h.update(array.tobytes())
                if epoch==1:
                    other=c.plan(seed,epoch,'balanced' if policy=='uniform' else 'uniform',source_labels,len(sets[2]))
                    assert all(np.array_equal(x,y) for x,y in zip((t,a,b),other[1:]))
            plans['%d_%s'%(seed,policy)]={'all_epoch_plan_sha256':h.hexdigest(),'planned_class_counts':total.tolist()}
    checks['all_1200_epoch_plans']=plans
    initial={}
    for seed in (2100,2101,2102):
        model,h=c.model(seed); initial[str(seed)]=h
        old=json.loads((c.base.OUT/('B_%d'%seed)/'config.json').read_text())
        assert h==old['initial_model_sha256']
    checks['initial_model_hashes_match_repaired_B']=initial
    # Real preprocessed input, actual source-only loss; target gradient must survive joint BN.
    gradients={}
    for norm,values in [('l2',sets),('zscore',zsets)]:
        model,_=c.model(2100); model.train()
        s,t,a,b=c.plan(2100,1,'uniform',source_labels,len(values[2]))
        sd=c.PlannedDataset(values[0]); td=c.PlannedDataset(values[2],True)
        source=torch.stack([sd[(int(i),int(k))][0] for i,k in zip(s[:8],a[:8])])
        target=torch.stack([td[(int(i),int(k))][0] for i,k in zip(t[:8],b[:8])]).requires_grad_()
        labels=torch.as_tensor(source_labels[s[:8]])
        assert source.shape==(8,1,102,13,13) and torch.isfinite(source).all()
        loss=torch.nn.functional.cross_entropy(model(source,target)[0],labels)
        loss.backward()
        target_l1=float(target.grad.abs().sum())
        assert target_l1>0 and torch.isfinite(loss) and torch.isfinite(target.grad).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradients[norm]={'source_ce':float(loss),'target_gradient_l1':target_l1,'parameters':sum(p.numel() for p in model.parameters())}
    checks['real_patch_joint_gradients']=gradients
    # Validate all six augmentation operations against native patch operators.
    for k in range(6):
        actual=c.PlannedDataset(sets[0])[(0,k)][0].numpy()[0].transpose(1,2,0)
        x,y=sets[0].indices[0]; patch=sets[0].data[x-6:x+7,y-6:y+7]
        expected=patch if k==0 else np.fliplr(patch) if k==1 else np.flipud(patch) if k==2 else np.rot90(patch,k=k-2)
        assert np.array_equal(actual,expected.astype(np.float32))
    checks['passed']=True; checks['seconds']=time.monotonic()-start
    c.dump(c.OUT/'verification.json',checks)
    print(json.dumps({'passed':True,'seconds':checks['seconds'],'path':str(c.OUT)}),flush=True)


if __name__=='__main__':main()
