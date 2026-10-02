"""Focused checks for BN semantics, domain routing, and gradients."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from normalization import canonical_state, install, set_domain

sys.path.insert(0, "/home/zhangzj26/IEEE_TCSVT_BiDA")
from models.get_model import get_model  # noqa: E402


def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    opts = SimpleNamespace(num_tokens=4, dim=64, depth=3)
    base = get_model("BiDA", "Houston13", 13, opts)
    source = torch.randn(2, 1, 48, 5, 5)
    target = torch.randn_like(source) * 2 + 3
    dsbn = copy.deepcopy(base)
    install(dsbn, "dsbn")
    state = torch.get_rng_state()
    base.train()
    original_logits = base(source, target)[0]
    original_logits.sum().backward()
    torch.set_rng_state(state)
    dsbn.train()
    dsbn_logits = dsbn(source, target)[0]
    dsbn_logits.sum().backward()
    assert torch.equal(original_logits, dsbn_logits)
    for (name, original), changed in zip(base.named_parameters(), dsbn.parameters()):
        assert (original.grad is None) == (changed.grad is None), name
        if original.grad is not None:
            assert torch.equal(original.grad, changed.grad), name
    assert dsbn.conv3d_features[1].source.weight is dsbn.conv3d_features[1].target.weight
    assert not torch.equal(dsbn.conv3d_features[1].source.running_mean,
                           dsbn.conv3d_features[1].target.running_mean)
    dsbn.eval()
    set_domain(dsbn, "source")
    source_eval = dsbn(source, source)[1]
    set_domain(dsbn, "target")
    target_eval = dsbn(source, source)[1]
    assert not torch.equal(source_eval, target_eval)
    print("DSBN: identical source logits/gradients, shared affine, distinct buffers and eval routes")

    left = torch.randn(3, 4, 5, 5)
    right = torch.randn_like(left) * 3 + 7
    joined = torch.cat((left, right))
    dims = (0, 2, 3)
    mu_s, mu_t = left.mean(dims), right.mean(dims)
    var_s, var_t = left.var(dims, unbiased=False), right.var(dims, unbiased=False)
    mix_mu = 0.5 * (mu_s + mu_t)
    mix_var = 0.5 * (var_s + var_t) + 0.25 * (mu_s - mu_t).square()
    torch.testing.assert_close(mix_mu, joined.mean(dims))
    torch.testing.assert_close(mix_var, joined.var(dims, unbiased=False))
    bn = nn.BatchNorm2d(4, momentum=0.19)
    bn(joined)
    torch.testing.assert_close(bn.running_mean, 0.19 * mix_mu)
    n = joined.numel() // 4
    torch.testing.assert_close(bn.running_var, 0.81 + 0.19 * mix_var * n / (n - 1))
    reverse = nn.BatchNorm2d(4, momentum=0.19)
    reverse(torch.cat((right, left)))
    torch.testing.assert_close(bn.running_mean, reverse.running_mean)
    torch.testing.assert_close(bn.running_var, reverse.running_var)

    fixed = get_model("BiDA", "Houston13", 13, opts)
    fixed.load_state_dict(base.state_dict())
    install(fixed, "fixed_mixture")
    fixed.zero_grad(set_to_none=True)
    target_grad = target.clone().requires_grad_(True)
    fixed.train()
    fixed(source, target_grad)[0].sum().backward()
    assert target_grad.grad is not None and target_grad.grad.abs().sum() > 0
    assert not fixed._paired_cache
    assert int(fixed.conv3d_features[1].num_batches_tracked) == int(base.conv3d_features[1].num_batches_tracked) + 1
    for arm, model in (("dsbn", dsbn), ("fixed_mixture", fixed)):
        restored = get_model("BiDA", "Houston13", 13, opts)
        install(restored, arm)
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        set_domain(model, "target")
        set_domain(restored, "target")
        with torch.no_grad():
            assert torch.equal(model(source, source)[1], restored(source, source)[1])
        assert set(canonical_state(restored)) == set(base.state_dict())
    print("Fixed mixture: exact pooled variance, order invariant buffers, target moment gradients, one update per pair")
    print("Both arms: strict checkpoint reload reproduces predictions")


if __name__ == "__main__":
    main()
