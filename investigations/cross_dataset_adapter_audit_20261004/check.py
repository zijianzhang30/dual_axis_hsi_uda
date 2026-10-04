"""Read-only audit of frozen runs; no training or historical edits."""
import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
NAS = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
native = NAS / 'tstnet_hyrank_native_v1/run_snapshot/datasets.py'
tree = ast.parse(native.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HyperX')
ns = dict(np=np, torch=torch)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(native), 'exec'), ns)
cls = ns['HyperX']
report = {'scope': 'Fresh CPU loader geometry and saved-prediction audit; no training',
          'flip_geometry': [], 'runs': {}}
for size in (12, 13):
    for horizontal, vertical in ((False, False), (True, False), (False, True), (True, True)):
        ds = cls.__new__(cls)
        ds.data = (np.arange(20)[:, None]*100 + np.arange(20)[None, :])[..., None]
        ds.label = np.ones((20, 20), dtype=int)
        ds.indices, ds.labels = [(10, 10)], [1]
        ds.patch_size, ds.flip_augmentation = size, True
        ds.radiation_augmentation = ds.mixture_augmentation = ds.center_pixel = False
        with patch.object(np.random, 'random', side_effect=[0., float(horizontal), float(vertical)]):
            image, label = ds[0]
        anchor = int(image[0, size//2, size//2])
        report['flip_geometry'].append(dict(patch=size, horizontal=horizontal, vertical=vertical,
                                            labeled_pixel=1010, actual_anchor_pixel=anchor,
                                            anchor_matches_labeled_pixel=anchor == 1010,
                                            returned_label=int(label)))
for group, arms in [('tstnet_pipeline_self_v1', ['B', 'C']),
                    ('tstnet_pipeline_bida_v1', ['A']),
                    ('tstnet_joint_adapt_2x2_v1', ['B_adapt', 'C_adapt'])]:
    for arm in arms:
        root = NAS/group/arm
        config = json.loads((root/'config.json').read_text())
        rows = [json.loads(x) for x in (root/'trajectory.jsonl').read_text().splitlines()]
        history = [json.loads(x) for x in (root/'history.jsonl').read_text().splitlines()]
        row = next(r for r in rows if r['epoch'] == 21)
        saved = np.load(root/'predictions/epoch_021.npz')
        truth, pred = saved['labels'].astype(int), saved['predictions'].astype(int)
        cm = np.bincount(12*truth+pred, minlength=144).reshape(12, 12)
        oa = np.trace(cm)/cm.sum()*100
        aa = (np.diag(cm)/cm.sum(1)).mean()*100
        assert abs(oa-row['oa']) < 1e-10 and abs(aa-row['aa']) < 1e-10
        report['runs'][arm] = dict(epoch21={'oa': oa, 'aa': aa, 'recalls': row['per_class_recall'],
                                          'zero_recall_classes': row['zero_recall_classes']},
                                  latest_evaluated={k: rows[-1][k] for k in ('epoch', 'oa', 'aa')},
                                  last_logged_epoch=history[-1]['epoch'],
                                  index_hashes=config['index_hashes'], labels_hashes=config['labels_hashes'],
                                  CE_pair_config_checks=config.get('CE_pair_config_checks'),
                                  mmd_batches_through21=[r.get('mmd_batches', 0) for r in history if r['epoch'] <= 21])
reference = report['runs']['B']
assert all(r['index_hashes'] == reference['index_hashes'] and r['labels_hashes'] == reference['labels_hashes']
           for r in report['runs'].values())
report['all_five_arms_same_centers_and_labels'] = True
report['probability_anchor_moves_under_native_flip_policy'] = 0.5 * 0.75
report['limits'] = ['Geometry defect is confirmed, its accuracy effect is not measured.',
                    'Epoch21 precedes MMD activation; its adaptation degradation cannot be caused by active MMD.',
                    'Latest evaluated epochs differ; use matched endpoints for comparison.']
(OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
