import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/zhangzj26/dual_axis_hsi_uda')
OLD = ROOT / 'results/mluda_pavia_native_v1'
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/mluda_mechanism_pavia_v1')
SEEDS = (1622, 1322, 1256)
ARMS = ('M0', 'M1', 'M2', 'M3')


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.dtype, a.shape)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def rng_hash():
    import random
    h = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())))
    h.update(torch.get_rng_state().numpy().tobytes())
    for rng in torch.cuda.get_rng_state_all():
        h.update(rng.cpu().numpy().tobytes())
    return h.hexdigest()
