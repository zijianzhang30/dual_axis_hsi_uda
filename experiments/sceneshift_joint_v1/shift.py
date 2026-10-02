"""Frozen historical noisy source-side SceneShift; no trainable parameters."""
import numpy as np
import torch
import torch.nn.functional as F

ALPHA=.7
CE_WEIGHT=.5
EPS=1e-5

def scene_stats(source_dataset,target_dataset,device):
    # BiDA loader reflection-pads by six; statistics must use the full,
    # normalized original cubes, including background but excluding padding.
    stats=[]
    for dataset in (source_dataset,target_dataset):
        cube=dataset.data[6:-6,6:-6,:]
        mean=cube.mean((0,1),dtype=np.float64).astype(np.float32)
        std=cube.std((0,1),ddof=0,dtype=np.float64).astype(np.float32)
        stats.extend([torch.from_numpy(mean).to(device)[None,None,:,None,None],
                      torch.from_numpy(std).to(device)[None,None,:,None,None]])
    return stats

def scene_shift(x,sm,ss,tm,ts,generator):
    # Adapt historical BCHW recipe to BiDA B1CHW without changing random
    # draw order or per-band affine/noise magnitudes.
    y=(x-sm)/(ss+EPS)
    y=y*((1-ALPHA)*ss+ALPHA*ts)+(1-ALPHA)*sm+ALPHA*tm
    scale=1+.04*torch.randn((len(x),1,1,1),generator=generator,device=x.device,dtype=x.dtype)
    noise=torch.randn(y[:,0].shape,generator=generator,device=x.device,dtype=x.dtype)
    return (y*scale[:,None]+.015*F.avg_pool2d(noise,5,1,2)[:,None]).clamp(0,1)
