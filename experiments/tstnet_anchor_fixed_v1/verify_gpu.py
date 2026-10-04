"""Check real tracked prefetch and strict metadata on student/EMA branches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import site
import sys
from types import SimpleNamespace

site.addsitedir('/home/zhangzj26/TGRS_MLUDA-2024/.venv/lib/python3.8/site-packages')
ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_anchor_fixed_v1')
sys.path.insert(0, str(ROOT / 'bida_snapshot'))
import numpy as np
import torch
import anchors
from models.get_model import get_model

torch.set_num_threads(2)
torch.manual_seed(1233)
np.random.seed(1233)
Dataset, Prefetch = anchors.tracked_loader_classes()
image = np.random.randn(28, 28, 176).astype(np.float32)
gt = np.zeros((28, 28), dtype=np.int64)
gt[8:20, 8:20] = 1
ds = Dataset(image, gt, patch_size=12, ignored_labels=[0], flip_augmentation=True,
             radiation_augmentation=True, mixture_augmentation=False,
             center_pixel=False, supervision='full')
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
stream = Prefetch(loader)
samples = 0
coords = []
for _ in range(len(loader)):
    x, y, position = stream.next()
    assert x.is_cuda and y.is_cuda and position.is_cuda
    assert position.shape == (len(x), 2)
    assert torch.all((position == 5) | (position == 6))
    samples += len(x)
    coords.extend(position.cpu().tolist())
assert samples == len(ds)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

joint = load('fixed_probe_joint', ROOT / 'frozen_components/normalization.py')
source = (ROOT / 'frozen_components/relation.py').read_text()
for a, b in [('bands=48', 'bands=176'), ('(1, 48)', '(1, 176)'), ('B×1×48×H×W', 'B×1×176×H×W')]:
    source = source.replace(a, b)
relation = {}
exec(compile(source, 'relation_176.py', 'exec'), relation)
models = []
for is_teacher in [False, True]:
    model = get_model('BiDA', 'Dioni', 12, SimpleNamespace(dim=64, depth=3, num_tokens=4), ema=is_teacher)
    joint.install(model, 'fixed_mixture')
    relation['install'](model, 'C', 1234 if is_teacher else 1233)
    before = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    anchors.install_model(model, allow_default=False)
    after = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    assert before == after
    models.append(model.cuda())
student, teacher = models
src = torch.randn(2, 1, 176, 12, 12, device='cuda')
tar = torch.randn_like(src).requires_grad_(True)
src_index = torch.tensor([[6, 5], [5, 6]], device='cuda')
tar_index = torch.tensor([[5, 5], [6, 6]], device='cuda')
anchors.set_context(student, src_index, tar_index)
student.train()
loss = torch.nn.functional.cross_entropy(student(src, tar)[0], torch.tensor([0, 1], device='cuda'))
loss.backward()
assert torch.isfinite(tar.grad).all() and tar.grad.abs().sum() > 0
anchors.set_context(teacher, src_index, tar_index)
with torch.no_grad():
    assert teacher(src, tar)[1].shape == (2, 12)
student.eval()
anchors.set_context(student, tar_index, tar_index)
with torch.no_grad():
    assert student(tar, tar)[1].shape == (2, 12)
try:
    student(tar, tar)
except RuntimeError as error:
    assert 'Missing loader anchor context' in str(error)
else:
    raise AssertionError('Missing metadata silently accepted')
report = dict(prefetch_samples=samples, observed_coordinates=sorted(set(tuple(c) for c in coords)),
              loss=float(loss), target_gradient_l1=float(tar.grad.abs().sum()), student_teacher_eval_interfaces=True,
              missing_metadata_rejected=True, parameter_buffers_unchanged_by_fix=True)
(ROOT / 'strict_gpu_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
