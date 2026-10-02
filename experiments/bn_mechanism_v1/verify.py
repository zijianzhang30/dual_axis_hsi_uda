"""Check forward identity and the intended gradient interventions."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from normalization import install, commit_buffers

sys.path.insert(0, '/home/zhangzj26/IEEE_TCSVT_BiDA')
from models.get_model import get_model

def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = get_model('BiDA', 'Houston13', 13, SimpleNamespace(num_tokens=4, dim=64, depth=3))
    original, balanced, full, detached = [copy.deepcopy(model) for _ in range(4)]
    install(balanced, 'buffer_only')
    install(detached, 'detach_target')
    spec = importlib.util.spec_from_file_location('joint_normalization', Path(__file__).resolve().parents[1] / 'bn_stabilization_v1/normalization.py')
    joint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joint)
    joint.install(full, 'fixed_mixture')
    source = torch.randn(2, 1, 48, 5, 5)
    target = torch.randn_like(source) * 2 + 3
    rng = torch.get_rng_state()
    outputs = []
    target_grads = []
    for arm in (original, balanced, full, detached):
        torch.set_rng_state(rng)
        xt = target.clone().requires_grad_(True)
        output = arm(source, xt)[0]
        output.square().sum().backward()
        outputs.append(output.detach())
        target_grads.append(xt.grad)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[2], outputs[3])
    for (name, a), b in zip(original.named_parameters(), balanced.parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name
    assert target_grads[2] is not None and target_grads[2].abs().sum() > 0
    assert target_grads[3] is None or target_grads[3].abs().sum() == 0
    for key, a in full.state_dict().items():
        assert torch.equal(a, detached.state_dict()[key]), key
    expected = {}
    for name, records in balanced._moment_records.items():
        (ms, vs, ns), (mt, vt, nt) = records
        expected[name] = ((ms + mt) / 2, ((vs + vt) / 2 + (ms - mt).square() / 4) * (ns + nt) / (ns + nt - 1))
    commit_buffers(balanced)
    for name, (mean, variance) in expected.items():
        bn = balanced.get_submodule(name)
        torch.testing.assert_close(bn.running_mean, .19 * mean)
        torch.testing.assert_close(bn.running_var, .81 + .19 * variance)
        assert int(bn.num_batches_tracked) == 1
    # Independent analytic check of the stopped-target moment gradient.
    xs = torch.randn(3, 4, 5, 5, dtype=torch.float64, requires_grad=True)
    xt = torch.randn_like(xs, requires_grad=True)
    bn = torch.nn.BatchNorm2d(4).double()
    native = bn(torch.cat((xs, xt.detach())))[:3]
    weight = torch.randn_like(native)
    (native * weight).sum().backward()
    native_grad = xs.grad.clone()
    xs.grad = None
    joined = torch.cat((xs, xt.detach()))
    mean = joined.mean((0, 2, 3), keepdim=True)
    var = joined.var((0, 2, 3), unbiased=False, keepdim=True)
    manual = (xs - mean) / (var + bn.eps).sqrt()
    (manual * weight).sum().backward()
    torch.testing.assert_close(native_grad, xs.grad, rtol=1e-10, atol=1e-10)
    assert xt.grad is None
    print('PASS: buffer-only logits/parameter gradients exact; balanced unbiased buffers correct')
    print('PASS: detached joint logits/buffers exact; target gradient stopped; analytic source gradient correct')

if __name__ == '__main__':
    main()
