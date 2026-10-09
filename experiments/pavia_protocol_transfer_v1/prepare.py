"""Lock all settings first; run actual official z-score and ILDA label-free."""
import ast
import shutil
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.decomposition import PCA
from skimage import exposure
import cv2
import common as c


def funcs(path, namespace, names=None):
    tree=ast.parse(path.read_text())
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and (names is None or x.name in names)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def sample_indices(labels,seed):
    # Exactly mirror get_sample_data: all class shuffles, concatenation, train AND val shuffles.
    rng=np.random.RandomState(seed)
    coords=np.argwhere(labels>0); y=labels[tuple(coords.T)]
    train=[];val=[]
    for k in range(1,8):
        ids=np.flatnonzero(y==k).tolist();rng.shuffle(ids)
        assert len(ids)>=180
        train.extend(ids[:180]);val.extend(ids[180:])
    rng.shuffle(train);rng.shuffle(val)
    return coords[train],y[train].astype(np.int64)-1


def augcodes(rng,n):
    codes=np.zeros(n,dtype=np.uint8)
    for i in range(n):
        if rng.random_sample()>.5: codes[i]=min(5,int(rng.random_sample()*5)+1)
    return codes


def main():
    assert not c.OUT.exists(),'Refuse overwrite'
    c.OUT.mkdir();(c.OUT/'inputs').mkdir();(c.OUT/'plans').mkdir()
    snap=c.OUT/'snapshot'
    shutil.copytree(c.P0/'snapshot',snap,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(c.P0/'scorer.py',snap/'scorer.py')
    (snap/'MLUDA').mkdir()
    for name in ('UtilsCMS.py','utils.py','config_UP2PC.py','MLUDA_up.py'):
        shutil.copy2(c.MLUDA/name,snap/'MLUDA'/name)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.txt'):shutil.copy2(path,c.OUT/path.name)
    fixed={'study':'Pavia frozen-C protocol transfer; no method innovation',
           'seeds':c.SEEDS,'new_arms':c.ARMS,'P0_reuse':str(c.P0/'pavia'),
           'bands':102,'classes':7,'patch':11,'anchor':[5,5],'batch':32,'epochs':100,
           'lr':.001,'momentum':.9,'weight_decay':.0005,'optimizer':'SGD recreated each epoch, matching official MLUDA',
           'steps_per_epoch':38,'source_samples':1260,'source_per_class':180,
           'step_rule':'MLUDA range(1,len(source_drop_last_loader)):38 of39 full batches;38x32=1216 samples/epoch',
           'padding':'official MLUDA constant zero for P1/P2',
           'augmentation':'preserve frozen C single-view hsi_augment: flip/rot90; no extra radiation views',
           'model':'frozen C: Full Joint BN(.19) + normalized center-relative nonlinear tokenizer + 3 self Transformer blocks',
           'tokens':4,'dim':64,'depth':3,'loss':'source-self CE only',
           'ILDA':{'pca_n':2,'epsilon':.00009,'guided_filter_radius':1,'preprocessing_rng_seed':2100,'train_label_access':False},
           'normalization':'official per-band scene-wise z-score over all pixels, separately source/target; before ILDA',
           'target_training':'same frozen P0 candidate coordinates, no target label values in training; historical nonzero-GT mask provenance disclosed',
           'evaluation':'same P0 evaluation pixels/order and target,target pairing; eval batch128/drop_last=False; fixed100 only',
           'target_best':False,'target_tuning':False,'no_additional_experiments':True,
           'rng':'precomputed private order/augmentation plans shared by P1/P2; independent scorer seed+41000',
           'P1_minus_P0_interpretation':'bundled protocol change, not attribution to any individual setting',
           'P2_minus_P1_interpretation':'official ILDA input adaptation only; not our innovation'}
    c.dump(c.OUT/'frozen_config.json',fixed)
    namespace={'np':np,'sio':__import__('scipy.io',fromlist=['loadmat']),'preprocessing':preprocessing}
    funcs(snap/'MLUDA/utils.py',namespace,{'load_data_pavia','get_sample_data'})
    s,sg=namespace['load_data_pavia'](str(c.DATA/'paviaU.mat'),str(c.DATA/'paviaU_gt_7.mat'))
    # Official target normalization does not need or read a target label file.
    raw=loadmat(c.DATA/'pavia.mat')['pavia']
    t=preprocessing.scale(raw.reshape(-1,raw.shape[-1])).reshape(raw.shape)
    assert s.shape[-1]==t.shape[-1]==102 and np.isfinite(s).all() and np.isfinite(t).all()
    audit={'source_shape':s.shape,'target_shape':t.shape,'zscore':{},'sampling':{}}
    for name,x in [('source',s),('target',t)]:
        f=x.reshape(-1,102)
        audit['zscore'][name]={'max_abs_band_mean':float(np.abs(f.mean(0)).max()),
                               'max_abs_band_std_error':float(np.abs(f.std(0)-1).max())}
        assert audit['zscore'][name]['max_abs_band_mean']<1e-8
        assert audit['zscore'][name]['max_abs_band_std_error']<1e-8
        np.save(c.OUT/'inputs'/('P1_%s.npy'%name),x.astype(np.float32))
    # Build P0 evaluation order once offline. Only coordinates/dummy labels go to training files.
    tg=loadmat(c.DATA/'pavia_gt_7.mat')['pavia_gt_7'].astype(np.int64)-1
    coords=np.argwhere(tg>=0)
    coords=coords[(coords[:,0]>0)&(coords[:,1]>0)]
    labels=tg[tuple(coords.T)]
    oldlabels=np.load(c.P0/'pavia/evaluation_labels.npy')
    assert np.array_equal(labels,oldlabels)
    oldcfg=__import__('json').loads((c.P0/'pavia/C_2100/config.json').read_text())
    assert c.digest(coords+6)==oldcfg['indices_sha256']['target_eval']
    assert oldcfg['indices_sha256']['target_eval']==oldcfg['indices_sha256']['target']
    np.save(c.OUT/'inputs/target_training_coords.npy',coords)
    np.save(c.OUT/'inputs/evaluation_labels_OFFLINE_ONLY.npy',labels)
    audit['evaluation']={'n':len(coords),'same_P0_order_and_labels':True,'coords_sha256':c.digest(coords)}
    for seed in c.SEEDS:
        pos,y=sample_indices(sg,seed)
        assert np.bincount(y,minlength=7).tolist()==[180]*7
        np.savez(c.OUT/'inputs'/('source_selection_%d.npz'%seed),coords=pos,labels=y)
        # Test exact source coordinates against actual official function on a tiny ID cube.
        state=np.random.get_state();np.random.seed(seed)
        ids=np.arange(s.shape[0]*s.shape[1],dtype=np.float64).reshape(*s.shape[:2],1)
        patches,ry=namespace['get_sample_data'](ids,sg,0,180)
        np.random.set_state(state)
        expected=pos[:,0]*s.shape[1]+pos[:,1]
        assert np.array_equal(patches[:,0,0,0],expected) and np.array_equal(ry,y)
        audit['sampling'][str(seed)]={'official_sampler_exact':True,'coords_sha256':c.digest(pos),'labels_sha256':c.digest(y)}
        plans={k:[] for k in ('source','target','source_aug','target_aug')}
        for epoch in range(1,101):
            order=np.random.RandomState(seed+epoch*100000+10000)
            source=order.permutation(1260)[:1216]
            target=np.random.RandomState(seed+epoch*100000+30000).permutation(len(coords))[:1216]
            sa=augcodes(np.random.RandomState(seed+epoch*100000+40000),1216)
            ta=augcodes(np.random.RandomState(seed+epoch*100000+50000),1216)
            for k,x in zip(plans,(source,target,sa,ta)):plans[k].append(x)
        np.savez_compressed(c.OUT/'plans'/('%d.npz'%seed),**{k:np.asarray(v) for k,v in plans.items()})
    # Actual official ILDA functions, unchanged; isolated NumPy RNG, no GT argument.
    ns={'np':np,'PCA':PCA,'exposure':exposure,'cv2':cv2}
    funcs(snap/'MLUDA/UtilsCMS.py',ns,{'pca','gammaCorrect','colorAdaption','GuideFilter','TotalAdaption','ILDA'})
    cv2.setNumThreads(2)
    rng=np.random.get_state();np.random.seed(2100)
    sa,ta=ns['ILDA'](s,t,2,.00009)
    np.random.set_state(rng)
    for name,x,ref in [('source',sa,s),('target',ta,t)]:
        assert x.shape==ref.shape and np.isfinite(x).all()
        np.save(c.OUT/'inputs'/('P2_%s.npy'%name),x.astype(np.float32))
    audit['ILDA']={'official_functions_unmodified':True,'labels_passed':False,'source_shape':sa.shape,'target_shape':ta.shape}
    audit['passed']=True;c.dump(c.OUT/'data_audit.json',audit)
    tracked=[p for p in c.OUT.rglob('*') if p.is_file() and p.suffix in ('.py','.npy','.npz','.txt')]
    tracked += [c.OUT/'frozen_config.json',c.OUT/'data_audit.json']
    c.dump(c.OUT/'provenance.json',{'file_sha256':{str(p.relative_to(c.OUT)):c.sha(p) for p in tracked},
        'input_mat_sha256':{str(c.DATA/name):c.sha(c.DATA/name) for name in ('paviaU.mat','paviaU_gt_7.mat','pavia.mat','pavia_gt_7.mat')}})
    print('Data/config locked; preflight required:',c.OUT,flush=True)


if __name__=='__main__': main()
