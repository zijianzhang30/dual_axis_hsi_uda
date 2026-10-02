"""Verify isolated order reversal and Fixed-Mixture/Joint-Batch equivalence."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from normalization import install

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/zhangzj26/IEEE_TCSVT_BiDA")
from models.get_model import get_model  # noqa: E402

spec = importlib.util.spec_from_file_location("fixed_bn", ROOT / "experiments/bn_stabilization_v1/normalization.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def joint_tokens(model, source, target):
    cube = model.conv3d_features(torch.cat((source, target), dim=0))
    b, c, bands, h, w = cube.shape
    feature = model.conv2d_features(cube.reshape(b, c * bands, h, w))
    tokens = model._forward_semantic_tokens(feature)
    return tokens.split(len(source), dim=0)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    opts = SimpleNamespace(num_tokens=4, dim=64, depth=3)
    initial = get_model("BiDA", "Houston13", 13, opts)
    source = torch.randn(2, 1, 48, 5, 5)
    target = torch.randn_like(source) * 2 + 3
    baseline, reverse = copy.deepcopy(initial), copy.deepcopy(initial)
    install(reverse, "reverse")
    moments = {}
    for name, layer in baseline.named_modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            def hook(module, inputs, name=name):
                value = inputs[0].detach()
                dims = (0,) + tuple(range(2, value.ndim))
                moments.setdefault(name, []).append((value.mean(dims), value.var(dims, unbiased=True)))
            layer.register_forward_pre_hook(hook)
    rng = torch.get_rng_state()
    outputs = baseline(source, target)
    outputs[0].square().sum().backward()
    torch.set_rng_state(rng)
    reversed_outputs = reverse(source, target)
    reversed_outputs[0].square().sum().backward()
    for left, right in zip(outputs, reversed_outputs):
        assert torch.equal(left, right)
    for (name, left), right in zip(baseline.named_parameters(), reverse.parameters()):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad), name
    for name, values in moments.items():
        (sm, sv), (tm, tv) = values
        rb = reverse.get_submodule(name)
        torch.testing.assert_close(rb.running_mean, 0.10 * sm + 0.09 * tm)
        torch.testing.assert_close(rb.running_var, 0.81 + 0.10 * sv + 0.09 * tv)
    print("Reverse-order: all branch logits and parameter gradients exact; buffers match target-then-source formula")

    separated, joint = copy.deepcopy(initial).double(), copy.deepcopy(initial).double()
    for model in (separated, joint):
        model.conv3d_features[1].momentum = 0.19
        model.conv2d_features[1].momentum = 0.19
    s1, t1 = source.double().requires_grad_(), target.double().requires_grad_()
    s2, t2 = source.double().requires_grad_(), target.double().requires_grad_()
    left = fixed.paired_tokens(separated, s1, t1)
    right = joint_tokens(joint, s2, t2)
    max_error = max(float((a - b).abs().max()) for a, b in zip(left, right))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    left[0].square().sum().backward()
    right[0].square().sum().backward()
    torch.testing.assert_close(s1.grad, s2.grad, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(t1.grad, t2.grad, rtol=1e-9, atol=1e-10)
    for (name, a), b in zip(separated.named_parameters(), joint.parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10)
    for name, a in separated.named_buffers():
        b = dict(joint.named_buffers())[name]
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    print(f"Fixed-Mixture equals independently implemented Joint-Batch: max token error={max_error:.3g}; input/parameter gradients and BN buffers agree")


if __name__ == "__main__":
    main()
