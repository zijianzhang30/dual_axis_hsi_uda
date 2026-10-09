"""Read-only HyRANK audit and epoch-500 clean evaluation; never trains."""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import site
import sys
from types import SimpleNamespace
from unittest.mock import patch

site.addsitedir('/home/zhangzj26/TGRS_MLUDA-2024/.venv/lib/python3.8/site-packages')
import numpy as np
import torch

RESULTS = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results')
OUT = RESULTS / 'hyrank_clean_audit_v1'
NATIVE = RESULTS / 'tstnet_hyrank_native_v1/run_snapshot'
FIX = RESULTS / 'tstnet_anchor_fixed_v1'
DATA = Path('/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK')
JOBS = {
    'BiDA': RESULTS / 'tstnet_pipeline_bida_v1/A',
    'B': RESULTS / 'tstnet_pipeline_self_v1/B',
    'C_fixed': FIX / 'C',
    'B_adapt': RESULTS / 'tstnet_joint_adapt_2x2_v1/B_adapt',
    'C_fixed_adapt': FIX / 'C_adapt',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def dataset_module():
    sys.path.insert(0, str(NATIVE))
    return load('clean_native_datasets', NATIVE / 'datasets.py')


def metrics(labels, pred):
    cm = np.bincount(12 * labels + pred, minlength=144).reshape(12, 12)
    recall = np.diag(cm) / cm.sum(1)
    shares = cm.sum(0) / cm.sum()
    observed = np.trace(cm) / cm.sum()
    expected = (cm.sum(0) * cm.sum(1)).sum() / float(cm.sum() ** 2)
    zero = (np.flatnonzero(recall == 0) + 1).tolist()
    return dict(oa=100 * observed, aa=100 * recall.mean(),
                kappa=(observed - expected) / (1 - expected),
                per_class_recall=(100 * recall).tolist(),
                prediction_shares=shares.tolist(), confusion_matrix=cm.tolist(),
                single_class_collapse=bool(shares.max() >= .95),
                zero_recall_class_count=len(zero), zero_recall_classes=zero)


def inventory():
    items = []
    for name, folder in JOBS.items():
        cfg = json.loads((folder / 'config.json').read_text())
        ckpt = folder / 'final_epoch500.pth'
        history = [json.loads(x) for x in (folder / 'history.jsonl').read_text().splitlines()]
        grid = [json.loads(x) for x in (folder / 'trajectory.jsonl').read_text().splitlines()]
        native = [r for r in grid if r['epoch'] == 500]
        items.append(dict(name=name, folder=str(folder), seed=cfg['seed'], trial=0,
                          checkpoint=str(ckpt), available=ckpt.exists(),
                          completed_epoch=history[-1]['epoch'],
                          config_sha256=sha(folder / 'config.json'),
                          native_epoch500=native[-1] if native else None))
    root = RESULTS / 'tstnet_hyrank_native_v1'
    # Ten sequential native trials: only a specifically saved final state is eligible.
    progress = [json.loads(x) for x in (root / 'progress.jsonl').read_text().splitlines()]
    for trial in range(10):
        ckpt = root / 'checkpoints' / ('trial_%02d_final.pth' % trial)
        p = [r for r in progress if r['trial'] == trial]
        items.append(dict(name='TSTnet', seed=1233, trial=trial,
                          trial_seed_note='Seed set once; subsequent trials continue native RNG',
                          checkpoint=str(ckpt), available=ckpt.exists(),
                          completed_epoch=p[-1]['epoch'] if p else None,
                          native_epoch500=None,
                          native_note='Native grid stops at 491; no native epoch500 evaluation'))
    write('inventory.json', items)
    return items


def summarize():
    items = inventory()
    evaluated = [json.loads(p.read_text()) for p in sorted(OUT.glob('*_epoch500.json'))]
    write('summary.json', dict(inventory=items, clean_results=evaluated,
        selection='All actual epoch500 weights; no target-best substitution; no higher-score selection',
        known_bug='Old C/C_adapt fixed (6,6) anchor under even-patch flips; excluded from corrected comparison',
        limitations=['Incomplete runs have no eligible checkpoint and are not evaluated',
                     'Native random test augmentation is an alternative protocol, not a declared bug',
                     'One seed1233 for BiDA/B/C tasks; TST trials are separate sequential runs',
                     'No claim that anchor correction explains the full accuracy gap']))
    source = Path(__file__).read_text()
    (OUT / 'code_snapshot.py').write_text(source)
    (OUT / 'minimal_code.diff').write_text(''.join(difflib.unified_diff([], source.splitlines(True),
          fromfile='/dev/null', tofile='investigations/hyrank_clean_audit_v1/check.py')))
    write('code_hashes.json', {str(Path(__file__)): sha(__file__)})


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    from scipy.io import loadmat
    ds = dataset_module()
    report = dict(normalization=[], data={}, configs={}, anchor_tests={})
    # Test the ACTUAL get_dataset return, not a reimplemented normalizer.
    for name in ['Dioni', 'Loukia']:
        image, gt, names, ignored, _, _ = ds.get_dataset(name, str(DATA) + '/')
        raw = loadmat(str(DATA / (name + '.mat')))['ori_data']
        real_gt = loadmat(str(DATA / (name + '_gt_out68.mat')))['map']
        assert image.shape[-1] == 176 and len(names) == 12
        assert np.array_equal(gt, real_gt) and set(np.unique(gt)) == set(range(13))
        norms = np.linalg.norm(image, axis=-1)
        assert float(abs(norms[norms > 0] - 1).max()) < 1e-6
        reference = raw.astype(np.float32)
        reference = reference / reference.max()
        denominator = np.sqrt((reference ** 2).sum(-1, keepdims=True))
        denominator[denominator == 0] = 1
        reference = reference / denominator
        returned_error = float(abs(reference - image).max())
        assert returned_error < 1e-6  # MAT/F-order and flattened reductions may differ by FP32 rounding.
        report['data'][name] = dict(shape=list(image.shape), names=names,
            gt_counts=np.bincount(gt.ravel().astype(int), minlength=13).tolist(),
            returned_unit_norm_max_error=float(abs(norms[norms > 0] - 1).max()), returned_vs_raw_formula_max_error=returned_error,
            mat_sha256=sha(DATA / (name + '.mat')), gt_sha256=sha(DATA / (name + '_gt_out68.mat')))
        del raw, image, reference
    synthetic = np.arange(5 * 7 * 176, dtype=np.float32).reshape(5, 7, 176) + 1
    for order in ['C', 'F']:
        cube = np.array(synthetic, order=order)
        fake_gt = np.ones((5, 7), dtype=np.int64)
        with patch.object(ds, 'open_file', side_effect=[{'ori_data': cube}, {'map': fake_gt}]):
            returned = ds.get_dataset('Dioni', str(DATA) + '/')[0]
        expected = synthetic.astype(np.float64) / np.linalg.norm(synthetic.astype(np.float64), axis=-1, keepdims=True)
        error = float(abs(returned - expected).max())
        assert error < 1e-6 and not np.array_equal(returned, cube)
        report['normalization'].append(dict(input_order=order, max_error=error, actual_return_checked=True))
    # Execute the existing verified tests in an isolated namespace, redirecting ONLY their report.
    for filename in ['verify.py', 'verify_gpu.py']:
        source = (FIX / filename).read_text()
        source = source.replace("ROOT / 'geometry_verification.json'", "Path(%r)" % str(OUT / 'geometry_verification.json'))
        source = source.replace("ROOT / 'strict_gpu_verification.json'", "Path(%r)" % str(OUT / 'strict_gpu_verification.json'))
        sys.path.insert(0, str(FIX))
        exec(compile(source, str(FIX / filename), 'exec'), {'__name__': '__main__'})
    report['anchor_tests'] = dict(geometry=json.loads((OUT / 'geometry_verification.json').read_text()),
                                 gpu=json.loads((OUT / 'strict_gpu_verification.json').read_text()))
    for name, folder in JOBS.items():
        cfg = json.loads((folder / 'config.json').read_text())
        report['configs'][name] = cfg
    index_hashes = [v['index_hashes']['test_dataset'] for v in report['configs'].values()]
    label_hashes = [v['labels_hashes']['test_dataset'] for v in report['configs'].values()]
    assert len(set(index_hashes)) == len(set(label_hashes)) == 1
    report['shared_native_test_order_hash'] = index_hashes[0]
    report['shared_native_test_label_hash'] = label_hashes[0]
    # Recreate constructor shuffles only, without iterating any loader or training batches.
    # get_dataset/sample_gt(split random_state23) do not consume NumPy RNG here.
    from utils_HSI import sample_gt
    source_gt = np.pad(loadmat(str(DATA / 'Dioni_gt_out68.mat'))['map'], ((7, 7), (7, 7)), 'constant')
    target_gt = np.pad(loadmat(str(DATA / 'Loukia_gt_out68.mat'))['map'], ((7, 7), (7, 7)), 'constant')
    selected_src = sample_gt(source_gt, .05)[0]
    selected_tar = sample_gt(target_gt, .05 * 10 * 20024 / 10317)[0]
    repeated = np.concatenate([selected_src] * 10)
    params = dict(patch_size=12, ignored_labels=[0], flip_augmentation=True, radiation_augmentation=True,
                  mixture_augmentation=False, center_pixel=False, supervision='full')
    np.random.seed(1233)
    for gt_for_init in [repeated, selected_tar]:
        ds.HyperX(SimpleNamespace(shape=gt_for_init.shape + (176,)), gt_for_init, **params)
    native_eval = ds.HyperX(SimpleNamespace(shape=target_gt.shape + (176,)), target_gt, **params)
    assert array_sha(native_eval.indices) == index_hashes[0]
    assert array_sha(native_eval.labels) == label_hashes[0]
    checked = []
    for name, folder in JOBS.items():
        first = folder / 'predictions/epoch_001.npz'
        if first.exists():
            saved = np.load(str(first))
            assert np.array_equal(saved['labels'], np.asarray(native_eval.labels) - 1)
            checked.append(name)
    first_tst = RESULTS / 'tstnet_hyrank_native_v1/predictions/trial_00_epoch_001.npz'
    if first_tst.exists():
        assert np.array_equal(np.load(str(first_tst))['labels'], np.asarray(native_eval.labels) - 1)
        checked.append('TSTnet/trial00')
    np.save(OUT / 'native_trial00_centers.npy', native_eval.indices)
    report['actual_native_predictions_label_order_checked'] = checked
    paths = [NATIVE / 'datasets.py', NATIVE / 'utils_HSI.py', FIX / 'anchors.py',
             FIX / 'C/derived_main.py', FIX / 'C_adapt/derived_main.py',
             RESULTS / 'tstnet_hyrank_native_v1/run.py',
             RESULTS / 'tstnet_pipeline_self_v1/source_snapshot/run.py',
             RESULTS / 'tstnet_pipeline_bida_v1/derived_main.py',
             RESULTS / 'tstnet_joint_adapt_2x2_v1/derived_main.py']
    report['dependency_hashes'] = {str(p): sha(p) for p in paths}
    report['active_entries'] = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().split(b'\0')
            command = [x.decode(errors='replace') for x in command if x]
            entries = [Path(x) for x in command if x.endswith('/run.py') and str(RESULTS) in x]
            if entries:
                report['active_entries'].append(dict(pid=int(proc.name), argv=command,
                  cwd=str((proc / 'cwd').resolve()), entry_hashes={str(p): sha(p) for p in entries}))
        except (OSError, PermissionError):
            continue
    # The CE baseline imports live BiDA while later variants import its frozen snapshot.
    # Clean evaluation may reuse the snapshot only after verifying the relevant files.
    comparisons = {}
    live = Path('/home/zhangzj26/IEEE_TCSVT_BiDA')
    for p in (FIX / 'bida_snapshot/models').rglob('*.py'):
        relative = p.relative_to(FIX / 'bida_snapshot')
        other = live / relative
        comparisons[str(relative)] = other.exists() and sha(p) == sha(other)
    for relative in ['normalization.py', 'relation.py']:
        other = Path('/home/zhangzj26/dual_axis_hsi_uda/experiments') / (
            'bn_stabilization_v1' if relative == 'normalization.py' else 'relation_tokenizer_v1') / relative
        comparisons[relative] = sha(FIX / 'frozen_components' / relative) == sha(other)
    assert all(comparisons.values()), 'Live/frozen inference dependency mismatch; report before evaluation'
    report['live_frozen_inference_dependency_checks'] = comparisons
    report['existing_fix_runtime_records'] = {}
    for name in ['C_fixed', 'C_fixed_adapt']:
        rows = [json.loads(x) for x in (JOBS[name] / 'history.jsonl').read_text().splitlines()]
        assert all(r['source_batch_trace_matches_original'] for r in rows)
        report['existing_fix_runtime_records'][name] = dict(epochs_checked=len(rows),
             last_anchor_counts=rows[-1]['actual_anchor_counts'], actual_source_trace_matches=True)
    report['scope'] = 'No training/code dependencies changed. Old wrong-anchor checkpoints excluded.'
    write('audit.json', report)
    inventory()


def clean_dataset():
    ds = dataset_module()
    image, gt, names, _, _, _ = ds.get_dataset('Loukia', str(DATA) + '/')
    image = np.pad(image, ((7, 7), (7, 7), (0, 0)), 'symmetric')
    gt = np.pad(gt, ((7, 7), (7, 7)), 'constant')
    import anchors
    Dataset, _ = anchors.tracked_loader_classes()
    dataset = Dataset(image, gt, patch_size=12, ignored_labels=[0], flip_augmentation=False,
                      radiation_augmentation=False, mixture_augmentation=False,
                      center_pixel=False, supervision='full')
    # Explicit canonical row-major order: same pixels for every model/trial, no shuffle.
    centers = np.column_stack(np.nonzero(gt))
    assert len(centers) == len(dataset) == 10317
    assert set(map(tuple, dataset.indices)) == set(map(tuple, centers))
    dataset.indices = centers
    dataset.labels = gt[centers[:, 0], centers[:, 1]].tolist()
    np.save(OUT / 'clean_centers.npy', centers)
    return dataset


def build(name):
    if name == 'TSTnet':
        from model import TSTnet
        return TSTnet.Feature_Extractor(176, 12, 12)
    sys.path.insert(0, str(FIX / 'bida_snapshot'))
    from models.get_model import get_model
    model = get_model('BiDA', 'Dioni', 12, SimpleNamespace(dim=64, depth=3, num_tokens=4))
    if name != 'BiDA':
        joint = load('clean_joint', FIX / 'frozen_components/normalization.py')
        joint.install(model, 'fixed_mixture')
    if name.startswith('C_fixed'):
        source = (FIX / 'frozen_components/relation.py').read_text()
        for a, b in [('bands=48', 'bands=176'), ('(1, 48)', '(1, 176)'), ('B×1×48×H×W', 'B×1×176×H×W')]:
            assert source.count(a) == 1
            source = source.replace(a, b)
        ns = {}
        exec(compile(source, 'clean_relation176.py', 'exec'), ns)
        ns['install'](model, 'C', 1233)
        import anchors
        anchors.install_model(model, allow_default=False)
    return model


def evaluate(name, trial):
    items = inventory()
    item = next(x for x in items if x['name'] == name and x['trial'] == trial)
    if not item['available']:
        print(json.dumps(dict(status='missing_actual_epoch500', **item)))
        return
    # No best/grid fallback allowed. State file named final was saved at completed epoch500.
    assert item['completed_epoch'] == 500 or name == 'TSTnet' and item['completed_epoch'] == 500
    sys.path.insert(0, str(FIX))
    dataset = clean_dataset()
    import anchors
    model = build(name).cuda().eval()
    checkpoint = Path(item['checkpoint'])
    checkpoint_hash = sha(checkpoint)
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu', weights_only=True), strict=True)
    before = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    runs = []
    for repeat in range(2):
        preds, truths = [], []
        with torch.inference_mode():
            for image, label, coordinate in loader:
                assert torch.all(coordinate == 6)
                image = image.cuda()
                if name == 'TSTnet':
                    logits = model(image)[0]
                else:
                    image = image.unsqueeze(1)
                    if name.startswith('C_fixed'):
                        coordinate = coordinate.cuda()
                        anchors.set_context(model, coordinate, coordinate)
                    logits = model(image, image)[1]
                assert logits.shape == (len(image), 12) and torch.isfinite(logits).all()
                preds.append(logits.argmax(1).cpu().numpy())
                truths.append(np.asarray(label) - 1)
        runs.append((np.concatenate(truths).astype(np.int64), np.concatenate(preds).astype(np.int64)))
    assert all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1]))
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
    assert all(not m.training for m in model.modules())
    assert sha(checkpoint) == checkpoint_hash
    truth, pred = runs[0]
    assert np.array_equal(truth, np.asarray(dataset.labels) - 1)
    result = dict(name=name, seed=item['seed'], trial=trial, epoch=500,
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  protocol='clean: all test augmentation disabled; fixed row-major pixel order',
                  centers_sha256=array_sha(dataset.indices), labels_sha256=array_sha(truth),
                  repeated_predictions_bitwise_equal=True, parameters_and_buffers_unchanged=True,
                  n=len(truth), metrics=metrics(truth, pred), native_epoch500=item['native_epoch500'])
    stem = '%s_trial%02d_epoch500' % (name, trial)
    np.savez_compressed(OUT / (stem + '.npz'), labels=truth, predictions=pred, centers=dataset.indices)
    write(stem + '.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit', 'inventory', 'eval', 'summary'])
    parser.add_argument('--name', choices=list(JOBS) + ['TSTnet'])
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.action == 'audit':
        audit()
    elif args.action == 'inventory':
        print(json.dumps(inventory(), indent=2))
    elif args.action == 'summary':
        summarize()
    else:
        if not args.name:
            parser.error('--name required for eval')
        evaluate(args.name, args.trial)
