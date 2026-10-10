"""Reuse the frozen diagnostic's math without changing its output directory."""
import importlib.util
import csv
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_scene_direction_math',HERE.parent/'common.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
ROOT=parent.ROOT
BASE=parent.OUT
OUT=BASE/'class_orthogonal_control'
SEEDS=parent.SEEDS
EPS=parent.EPS
PROBES=parent.PROBES
REPS=parent.REPS
sha=parent.sha
ahash=parent.ahash
dump=parent.dump
canonical=parent.canonical
exact_attack=parent.exact_attack
metric=parent.metric
aggregate=parent.aggregate

CONFIG={'source_class_subspace':'SVD of 64x7 class-centroid-minus-global-centroid matrix, numeric rank capped6',
    'source_population':'all2520 original source rows, same as parent',
    'projection':'remove class-basis projection, SVD/QR orthonormalize; no replacement directions',
    'rank':'same as original domain probe:1 or4, unless projected rank degenerates',
    'random':'reuse all16 parent random bases, project into SAME class-orthogonal space, orthonormalize',
    'epsilon':list(EPS),'attack':'parent exact source-label-guided worst true-vs-competitor margin rule',
    'budget':'epsilon times original per-source rawZ L2 norm, no renormalization after attack',
    'criterion':parent.CONFIG['clear_margin_rule'],
    'decision':'No-Go for robustness direction if neither projected probe passes all3seeds/all3eps',
    'target_GT_never_accessed':True,'no_training_backprop_tuning_followup':True}

def csvwrite(name,rows):
    with (OUT/name).open('x',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def projected_basis(b,class_basis):
    residual=b-class_basis@(class_basis.T@b)
    u,s,vh=np.linalg.svd(residual,full_matrices=False)
    tol=np.finfo(float).eps*max(residual.shape)*max(1.,float(s[0]))
    rank=int((s>tol).sum());requested=b.shape[1]
    if rank<requested:return None,residual,s,rank
    q=u[:,:requested]
    q=q-class_basis@(class_basis.T@q)
    q=canonical(np.linalg.qr(q,mode='reduced')[0])
    assert np.max(abs(q.T@q-np.eye(requested)))<1e-12
    assert np.max(abs(class_basis.T@q))<1e-10
    return q,residual,s,rank
