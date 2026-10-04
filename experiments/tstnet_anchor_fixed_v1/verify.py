"""CPU flip geometry, native tensor/RNG parity and relation equivariance."""
import ast
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import anchors

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_anchor_fixed_v1')
ROOT.mkdir(parents=True, exist_ok=True)
tree = ast.parse(anchors.NATIVE.read_text())
native_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HyperX')
ns = dict(np=np, torch=torch)
exec(compile(ast.Module(body=[native_node], type_ignores=[]), str(anchors.NATIVE), 'exec'), ns)
Native = ns['HyperX']
Tracked, _ = anchors.tracked_loader_classes()


def stub(cls, size, noise=False):
    ds = cls.__new__(cls)
    ds.data = (np.arange(20)[:, None] * 100 + np.arange(20)[None, :])[..., None].astype(np.float32)
    ds.label = np.ones((20, 20), dtype=np.int64)
    ds.indices, ds.labels = [(10, 10)], [1]
    ds.patch_size, ds.flip_augmentation = size, True
    ds.radiation_augmentation = noise
    ds.mixture_augmentation = ds.center_pixel = False
    return ds


def same_rng(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


torch.set_num_threads(2)
report = {'geometry': [], 'native_tensor_and_rng_replays': 0}
for size in [12, 13]:
    for horizontal, vertical in [(False, False), (True, False), (False, True), (True, True)]:
        with patch.object(np.random, 'random', side_effect=[0., float(horizontal), float(vertical)]):
            image, label, index = stub(Tracked, size)[0]
        actual = float(image[0, index[0], index[1]])
        assert actual == 1010 and label == 1
        report['geometry'].append(dict(patch=size, horizontal=horizontal, vertical=vertical, anchor=index.tolist(), pixel=actual))
        with patch.object(np.random, 'random', side_effect=[0., float(horizontal), float(vertical)]):
            old_image, old_label = stub(Native, size)[0]
        assert torch.equal(image, old_image) and int(label) == int(old_label)
    for seed in range(100):
        np.random.seed(seed)
        start = np.random.get_state()
        a = stub(Native, size, True)[0]
        end = np.random.get_state()
        np.random.set_state(start)
        b = stub(Tracked, size, True)[0]
        assert torch.equal(a[0], b[0]) and int(a[1]) == int(b[1])
        assert same_rng(end, np.random.get_state())
        report['native_tensor_and_rng_replays'] += 1

# Real constructor/indices verify the AST class-name substitution, not just __new__.
params = dict(patch_size=12, ignored_labels=[0], flip_augmentation=True, radiation_augmentation=True,
              mixture_augmentation=False, center_pixel=False, supervision='full')
image = np.zeros((25, 25, 176), dtype=np.float32)
gt = np.zeros((25, 25), dtype=np.int64)
gt[8:18, 8:18] = 1
np.random.seed(1233)
start = np.random.get_state()
old_ds = Native(image, gt, **params)
end = np.random.get_state()
np.random.set_state(start)
new_ds = Tracked(image, gt, **params)
assert np.array_equal(old_ds.indices, new_ds.indices) and old_ds.labels == new_ds.labels
assert same_rng(end, np.random.get_state())
report['constructor_indices_labels_rng_identical'] = True

source = Path('/home/zhangzj26/dual_axis_hsi_uda/experiments/relation_tokenizer_v1/relation.py').read_text()
for a, b in [('bands=48', 'bands=176'), ('(1, 48)', '(1, 176)'), ('B×1×48×H×W', 'B×1×176×H×W')]:
    source = source.replace(a, b)
rel = {}
exec(compile(source, 'relation_176.py', 'exec'), rel)
torch.manual_seed(1233)
score = rel['RelationScores']('C')
patches = torch.randn(3, 1, 176, 12, 12)
reference = score(patches)
from types import MethodType
score._anchor_allow_default = False
score.relations = MethodType(anchors.relations, score)
score.forward = MethodType(anchors.scores, score)
anchor = torch.tensor([[6, 6]]).expand(3, 2)
assert torch.equal(reference, score(patches, anchor))
errors = []
for horizontal, vertical in [(False, False), (True, False), (False, True), (True, True)]:
    dims = ([-1] if horizontal else []) + ([-2] if vertical else [])
    moved = torch.flip(patches, dims) if dims else patches
    coordinate = torch.tensor([[5 if vertical else 6, 5 if horizontal else 6]]).expand(3, 2)
    actual = score(moved, coordinate).reshape(3, 4, 12, 12)
    expected = reference.reshape(3, 4, 12, 12)
    if dims:
        expected = torch.flip(expected, dims)
    error = float((actual - expected).abs().max())
    assert error < 1e-6
    errors.append(error)
report['unflipped_scores_exactly_identical'] = True
report['flip_equivariance_max_errors'] = errors
(ROOT / 'geometry_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
