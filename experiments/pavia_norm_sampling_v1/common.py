"""Independent streams and native Pavia model/data semantics."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import scale

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_norm_sampling_v1')
spec = importlib.util.spec_from_file_location('sampling_fixed_base', ROOT/'experiments/pavia_l2_fixed_v1/common.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
helper = base.helper
N = 128 * 290
ARMS = {'P0': ('l2', 'uniform'), 'P1': ('zscore', 'uniform'),
        'P2': ('l2', 'balanced'), 'P3': ('zscore', 'balanced')}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def scene(name, norm):
    directory = base.DATA/'Pavia'
    if norm == 'l2':
        return base.load_scene(directory, name)
    raw = loadmat(directory/(name+'.mat'))[name]
    image = scale(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape)
    labels = loadmat(directory/(name+'_gt_7.mat'))[name+'_gt_7'].astype(int)-1
    assert image.shape[-1] == 102 and np.isfinite(image).all()
    return image, labels


def datasets(seed, norm):
    source, sg = scene('paviaU', norm)
    target, tg = scene('pavia', norm)
    train_gt, val_gt = base.pavia.sample_gt(sg, .95, seed, mode='random')
    target_gt, _ = base.pavia.sample_gt(tg, 1, seed, mode='random')
    pad = lambda x: np.pad(x, ((6,6),(6,6),(0,0)), mode='reflect')
    pg = lambda x: np.pad(x, ((6,6),(6,6)), mode='reflect')
    source, target = pad(source), pad(target)
    cls = base.pavia.HSIDataset
    values = [cls(source, pg(train_gt), 13, data_aug=False),
              cls(source, pg(val_gt), 13, data_aug=False),
              cls(target, pg(target_gt), 13, data_aug=False)]
    assert [len(x) for x in values] == [37236,1960,39348]
    return values


def codes(seed, epoch, offset):
    rng = np.random.RandomState(seed + offset + 100000*epoch)
    result = np.zeros(N, dtype=np.uint8)
    # Consume the same conditional draws as native hsi_augment, privately.
    for i in range(N):
        if rng.random_sample() > .5:
            result[i] = min(5, int(rng.random_sample()*5)+1)
    return result


def plan(seed, epoch, policy, labels, ntarget):
    counts = np.bincount(labels, minlength=7)
    weights = np.ones(len(labels)) if policy == 'uniform' else 1./counts[labels]
    generator = torch.Generator().manual_seed(seed + (10000 if policy == 'uniform' else 20000) + 100000*epoch)
    src = torch.multinomial(torch.as_tensor(weights, dtype=torch.double), N,
                            replacement=True, generator=generator).numpy()
    tg = torch.Generator().manual_seed(seed+30000+100000*epoch)
    target = torch.randperm(ntarget, generator=tg)[:N].numpy()
    return src, target, codes(seed,epoch,40000), codes(seed,epoch,50000)


class PlannedDataset(torch.utils.data.Dataset):
    def __init__(self, native, unlabeled=False):
        self.native, self.unlabeled = native, unlabeled

    def __len__(self):
        return len(self.native)

    def __getitem__(self, token):
        index, code = token
        x, y = self.native.indices[index]
        patch = self.native.data[x-6:x+7, y-6:y+7]
        if code == 1:
            patch = np.fliplr(patch)
        elif code == 2:
            patch = np.flipud(patch)
        elif code >= 3:
            patch = np.rot90(patch, k=code-2)
        value = torch.from_numpy(np.asarray(np.copy(patch).transpose(2,0,1), dtype=np.float32)).unsqueeze(0)
        label = 0 if self.unlabeled else int(self.native.labels[index])
        return value, label, index, code


class DrawSampler(torch.utils.data.Sampler):
    def __init__(self):
        self.indices = self.augmentation = None

    def __len__(self):
        return N

    def __iter__(self):
        return iter(zip(self.indices.tolist(), self.augmentation.tolist()))


def model(seed):
    base.pavia.seed_worker(seed)
    return base.build('pavia','B',seed)


def metric(labels, predictions, logits):
    m = helper.metrics(labels,predictions,logits,7)
    cm = np.asarray(m['confusion_matrix'])
    m.update(single_class_collapse=max(m['prediction_shares']) >= .95,
             zero_recall_classes=[i+1 for i,x in enumerate(m['per_class_percent']) if x == 0])
    m['zero_recall_class_count'] = len(m['zero_recall_classes'])
    for i in (2,3):
        m['error_%d_to_7_count'%(i+1)] = int(cm[i,6])
        m['error_%d_to_7_ratio'%(i+1)] = float(cm[i,6]/cm[i].sum())
    return m
