"""Independent short numerical audit; does not alter historical experiments."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

ROOT = Path('/home/zhangzj26/dual_axis_hsi_uda')
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/pavia_transfer_audit_v1')
OFFICIAL = OUT.parent / 'hyrank_repro_audit_v2/dependency_snapshot/official_BiDA_pinned'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


c = module('transfer_extension_common', ROOT / 'experiments/mluda_bida_extension_v1/common.py')


def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())


def restore(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3])


def compare(left, right, name, report, tolerance=0.):
    assert set(left) == set(right), name
    maximum = 0.
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None:
            assert a is None and b is None, (name, key, 'gradient presence')
            continue
        error = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
        maximum = max(maximum, error)
        if not torch.allclose(a, b, atol=tolerance, rtol=tolerance):
            raise AssertionError((name, key, error, 'first divergence'))
    report[name] = maximum


def step(model, optimizer, x, y, target):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(x, target)
    loss = F.cross_entropy(output[0], y)
    loss.backward()
    gradients = {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}
    optimizer.step()
    return ({str(i): value.detach().clone() for i, value in enumerate(output)},
            {'ce': loss.detach()}, gradients,
            {k: value.clone() for k, value in model.state_dict().items()})


def replay(left, right, x, y, target, report, prefix, tolerance=0., n=3):
    a = torch.optim.SGD(left.parameters(), lr=.01)
    b = torch.optim.SGD(right.parameters(), lr=.01)
    for i in range(n):
        state = rng()
        l = step(left, a, x, y, target)
        post = rng()
        restore(state)
        r = step(right, b, x, y, target)
        assert torch.equal(post[2], torch.get_rng_state())
        assert torch.equal(post[3], torch.cuda.get_rng_state())
        for name, av, bv in zip(('logits', 'source_ce', 'gradients', 'updated_state'), l, r):
            compare(av, bv, '%s_step%d_%s' % (prefix, i+1, name), report, tolerance)
    left.eval(); right.eval()
    with torch.no_grad():
        compare({'target': left(target, target)[1]}, {'target': right(target, target)[1]},
                prefix + '_eval_logits', report, tolerance)


def install_reference_joint(model):
    """Independent full concatenation at each convolution/BN, native remainder."""
    native_forward = model.forward.__func__
    for seq in (model.conv3d_features, model.conv2d_features):
        seq[1].momentum = .19
    native_tokenize = model._tokenize.__func__
    def tokens(self, image):
        if self.training:
            return self.reference_tokens.pop(0)
        return native_tokenize(self, image)
    def forward(self, source, target):
        if self.training:
            z = self.conv3d_features(torch.cat((source, target), 0))
            z = z.flatten(1, 2)
            z = self.conv2d_features(z)
            sf, tf = z.chunk(2, 0)
            self.reference_tokens = [self._forward_semantic_tokens(sf), self._forward_semantic_tokens(tf)]
        result = native_forward(self, source, target)
        if self.training:
            assert not self.reference_tokens
        return result
    model._tokenize = MethodType(tokens, model)
    model.forward = MethodType(forward, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=False)
    shutil.copy2(__file__, OUT / 'verify.py')
    shutil.copy2(Path(__file__).with_name('PROTOCOL.md'), OUT / 'PROTOCOL.md')
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    device = torch.device('cuda:%d' % args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = {'scope': 'Pavia adapted self CE only; not full BiDA/EMA numerical certification',
              'official_commit': '66a92b542ffa1df3c8b628af3d8bd001b799f1f6',
              'float32_joint_tolerance': 5e-4, 'device': str(device), 'errors': {}}
    try:
        official = module('transfer_pinned_official_model', OFFICIAL / 'models/BiDA.py')
        assert c.helper.file_hash(OFFICIAL / 'models/BiDA.py') == c.helper.file_hash(c.pavia.BIDA / 'models/BiDA.py')
        c.pavia.seed_worker(2100)
        values = c.loaders('pavia', 2100, 0)
        audit = c.inputs_audit('pavia', values)
        # Extract the exact generic upstream normband operator without adding Pavia
        # to the upstream dataset dispatch or importing another training entry.
        tree = ast.parse((OFFICIAL / 'utils/dataset.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_mat_hsi')
        wanted = ast.dump(ast.parse("norm == 'normband'", mode='eval').body)
        branch = next(n for n in fn.body if isinstance(n, ast.If) and ast.dump(n.test) == wanted)
        normalization = compile(ast.Module(body=branch.body, type_ignores=[]), '<native_normband>', 'exec')
        for name in ('paviaU', 'pavia'):
            raw = loadmat(c.DATA / 'Pavia' / (name + '.mat'))[name]
            ns = {'np': np, 'img': raw.copy()}
            exec(normalization, ns)
            local, _ = c.pavia.load_scene(c.DATA / 'Pavia', name)
            error = float(np.abs(ns['img'] - local).max())
            assert error == 0., (name, error)
            report[name + '_normalization_max_error'] = error
        # Reconstruct the archived upstream dataset with its sole necessary runtime
        # repair self.data=image. Compare augmented patches under identical RNG.
        original_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HSIDataset')
        init = next(n for n in original_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        init.body.insert(0, ast.parse('self.data = image').body[0])
        namespace = dict(sys.modules[c.pavia.HSIDataset.__module__].__dict__)
        ast.fix_missing_locations(original_cls)
        exec(compile(ast.Module(body=[original_cls], type_ignores=[]), '<native_dataset_one_fix>', 'exec'), namespace)
        for which in (0, 2):
            ds = values[which].dataset
            if hasattr(ds, 'original'): ds = ds.original
            independent = namespace['HSIDataset'](ds.data, ds.label, 13, data_aug=True)
            assert np.array_equal(independent.indices, ds.indices)
            for index in (0, 123, len(ds)-1):
                state = rng(); left = independent[index]; restore(state); right = ds[index]
                assert torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])
        report['native_augmented_patch_comparison'] = 'exact for six fixed positions, identical RNG'
        x, y = next(iter(values[0])); t, dummy = next(iter(values[2]))
        assert x.shape == t.shape == (128, 1, 102, 13, 13)
        assert (dummy == 0).all()
        x, y, t = x.to(device), y.to(device), t.to(device)
        c.pavia.seed_worker(2100)
        local = c.pavia.BiDAnet(n_bands=102, num_classes=7, num_tokens=4, dim=64, depth=3).to(device)
        native = official.BiDAnet(n_bands=102, num_classes=7, num_tokens=4, dim=64, depth=3).to(device)
        native.load_state_dict(copy.deepcopy(local.state_dict()))
        replay(native, local, x, y, t, report['errors'], 'native_vs_local', n=3)
        del native, local
        c.pavia.seed_worker(2100)
        routed, _ = c.build('pavia', 'B', 2100)
        independent = official.BiDAnet(n_bands=102, num_classes=7, num_tokens=4, dim=64, depth=3)
        independent.load_state_dict(copy.deepcopy(routed.state_dict()))
        install_reference_joint(independent)
        replay(independent.to(device), routed.to(device), x, y, t, report['errors'],
               'explicit_joint_vs_routed', tolerance=5e-4, n=3)
        del independent, routed
        gradients = {}
        for arm in ('B', 'C'):
            c.pavia.seed_worker(2100)
            model, _ = c.build('pavia', arm, 2100)
            model.to(device).train()
            if arm == 'C':
                relation = model.relation_scores
                expected = (x[:,0] - x[:,0,:,6:7,6:7]) / (x[:,0].std((-2,-1), correction=0, keepdim=True) + 1e-6)
                assert torch.equal(relation.relations(x), expected)
                weights = relation(x)
                assert weights.shape == (128,4,169) and torch.isfinite(weights).all()
                assert torch.allclose(weights.sum(-1), torch.ones(128,4,device=device), atol=1e-6)
                assert torch.isfinite(relation(torch.ones_like(x))).all()
                native_semantic = model._forward_semantic_tokens.__func__
                captured = []
                def check_tokens(self, features):
                    patch = self._relation_pending[0]
                    result = native_semantic(self, features)
                    expected_tokens = torch.einsum('bln,bcn->blc', self.relation_scores(patch), features.flatten(2))
                    assert features.shape[-2:] == patch.shape[-2:]
                    assert torch.equal(result, expected_tokens)
                    captured.append(tuple(features.shape))
                    return result
                model._forward_semantic_tokens = MethodType(check_tokens, model)
            target = t.detach().clone().requires_grad_(True)
            loss = F.cross_entropy(model(x,target)[0], y)
            loss.backward()
            assert target.grad is not None and torch.isfinite(target.grad).all()
            assert target.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if arm == 'C':
                assert len(captured) == 2
                assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.relation_scores.parameters())
            gradients[arm] = float(target.grad.abs().sum())
            del model, target, loss
        report['target_input_gradient_l1_from_source_ce'] = gradients
        report['data_protocol'] = audit
        report['passed'] = True
        c.dump(OUT / 'verification.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(passed=False, first_divergence=repr(error))
        c.dump(OUT / 'verification.json', report)
        raise


if __name__ == '__main__': main()
