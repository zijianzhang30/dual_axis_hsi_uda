"""Frozen C architecture, explicit Pavia protocol-transfer inputs and streams."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode = True
PROJECT = Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_protocol_transfer_v1')
P0 = OUT.parent / 'tokenizer_mechanism_pavia_l2_fixed_v1'
MLUDA = OUT.parent / 'mluda_pavia_native_v1/native_snapshot'
DATA = Path('/home/zhangzj26/TGRS_MLUDA-2024/datasets/Pavia')
SEEDS = (2100, 2101, 2102)
ARMS = ('P1', 'P2')


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for v in model.state_dict().values():
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def environment():
    snap = OUT/'snapshot'
    sys.path.insert(0, str(snap/'BiDA/experiments/bida_memory_v1'))
    sys.path.insert(0, str(snap/'BiDA'))
    from models.BiDA import BiDAnet
    from utils.utils_HSI import seed_worker
    return BiDAnet, seed_worker, module('protocol_joint',snap/'normalization.py'), module('protocol_metrics',snap/'helpers.py')


def build(seed):
    BiDAnet, seed_worker, joint, _ = environment()
    seed_worker(seed)
    model = BiDAnet(n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3)
    unused = BiDAnet(n_bands=102,num_classes=7,num_tokens=4,dim=64,depth=3)
    del unused
    original = state_hash(model)
    joint.install(model,'fixed_mixture')
    scorer = module('protocol_scorer',OUT/'snapshot/scorer.py')
    cpu = torch.get_rng_state().clone()
    scorer.install(model,'C',seed,102)
    assert torch.equal(cpu,torch.get_rng_state())
    reference = json.loads((P0/'pavia'/('C_%d'%seed)/'config.json').read_text())
    assert original == reference['initial_original_model_sha256']
    assert state_hash(model) == reference['initial_full_model_sha256']
    return model, original


def config(seed, arm):
    locked = json.loads((OUT/'frozen_config.json').read_text())
    return dict(locked,seed=seed,arm=arm,input='per-band z-score'+(' + official ILDA' if arm=='P2' else ''))


def augment(patch, code):
    if code == 1: patch = np.fliplr(patch)
    elif code == 2: patch = np.flipud(patch)
    elif code >= 3: patch = np.rot90(patch,k=code-2)
    return patch


class PatchSet(torch.utils.data.Dataset):
    def __init__(self, arm, domain, coords, labels=None):
        cube = np.load(OUT/'inputs'/('%s_%s.npy'%(arm,domain)),mmap_mode='r')
        self.data = np.pad(cube,((5,5),(5,5),(0,0)),mode='constant')
        self.coords = np.asarray(coords,dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels,dtype=np.int64)

    def __len__(self): return len(self.coords)

    def __getitem__(self, key):
        index, code = key if isinstance(key,tuple) else (key,0)
        row,col = self.coords[index]
        patch = augment(self.data[row:row+11,col:col+11],int(code))
        x = torch.from_numpy(np.array(patch.transpose(2,0,1),dtype=np.float32,copy=True)).unsqueeze(0)
        return x, 0 if self.labels is None else int(self.labels[index])


class PlanSampler(torch.utils.data.Sampler):
    def __init__(self, indices, codes): self.indices,self.codes=indices,codes
    def __len__(self): return len(self.indices)
    def __iter__(self): return iter(zip(self.indices.tolist(),self.codes.tolist()))


def train_sets(arm, seed):
    with np.load(OUT/'inputs'/('source_selection_%d.npz'%seed)) as a:
        source=PatchSet(arm,'source',a['coords'],a['labels'])
    target=PatchSet(arm,'target',np.load(OUT/'inputs/target_training_coords.npy'))
    return source,target


def epoch_loaders(source,target,seed,epoch):
    with np.load(OUT/'plans'/('%d.npz'%seed)) as a:
        arrays=[a[k][epoch-1] for k in ('source','target','source_aug','target_aug')]
    kwargs=dict(batch_size=32,num_workers=0,drop_last=False)
    # Private generators prevent DataLoader iterator setup from changing model/dropout RNG.
    loaders=[torch.utils.data.DataLoader(ds,sampler=PlanSampler(i,c),
                 generator=torch.Generator().manual_seed(seed+offset+epoch),**kwargs)
             for ds,i,c,offset in [(source,arrays[0],arrays[2],81000),(target,arrays[1],arrays[3],82000)]]
    return loaders,arrays


def verify_frozen():
    p=json.loads((OUT/'provenance.json').read_text())
    for rel,h in p['file_sha256'].items(): assert sha(OUT/rel)==h,rel
    return p
