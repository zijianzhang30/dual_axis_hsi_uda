import numpy as np,json
from argparse import Namespace
from common import OUT,BASE,frozen,dump,hasharray,sha,SEEDS
assert not (OUT/'data').exists();(OUT/'data').mkdir();env=frozen.environment();env[4](2100);args=Namespace(dataset='houston',dataset_dir='/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston/',seed=2100,num_workers=0);ls=frozen.loaders(args)
from utils.dataset import open_file
scoords=np.unique(np.concatenate([ls[0].dataset.indices,ls[1].dataset.indices]),axis=0);tcoords=ls[3].dataset.indices;trainset=set(map(tuple,ls[0].dataset.indices));trainmask=np.array([tuple(v) in trainset for v in scoords]);assert trainmask.sum()==2393 and (~trainmask).sum()==127
record={}
for domain,coords,cube,scene in [('source',scoords,ls[0].dataset.data,'Houston13'),('target',tcoords,ls[3].dataset.data,'Houston18')]:
 gt=open_file(args.dataset_dir+scene+'_7gt.mat')['map'].astype(np.int16)-1;shape=gt.shape;gt=np.pad(gt,6,mode='reflect');labels=gt[coords[:,0],coords[:,1]]
 if domain=='target':assert np.array_equal(labels,np.load(BASE/'houston/evaluation_labels.npy'))
 for name,a in [('cube',cube),('coords',coords),('gt',gt),('labels',labels)]:np.save(OUT/'data'/f'{domain}_{name}.npy',a)
 rng=np.random.default_rng(97000);ids=rng.choice(len(coords),min(4096,len(coords)),replace=False);values=cube[coords[ids,0],coords[ids,1]];norm=np.linalg.norm(values,axis=1);assert np.max(np.abs(norm-1))<1e-5
 record[domain]={'samples':len(coords),'shape':list(shape),'coords_sha256':hasharray(coords),'cube_sha256':hasharray(cube),'labels_sha256':hasharray(labels),'pixel_spectral_L2_norm_mean':float(norm.mean()),'pixel_spectral_L2_norm_max_error':float(abs(norm-1).max())}
np.save(OUT/'data/source_probe_train_mask.npy',trainmask)
dump(OUT/'fixed_inputs.json',{'domains':record,'source_probe_train':2393,'source_probe_heldout':127,'fixed_probe_partition_seed':2100,'checkpoints':{str(seed):{'path':str(BASE/'houston'/f'C_{seed}'/'checkpoints/epoch_200.pth'),'sha256':sha(BASE/'houston'/f'C_{seed}'/'checkpoints/epoch_200.pth')} for seed in SEEDS}});print('Fixed data prepared')
