"""Check the query transplant under independent joint-stem execution."""
import copy
from types import SimpleNamespace
import torch
import train

def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    base = train.get_model('BiDA', 'Houston13', 13, SimpleNamespace(num_tokens=4, dim=64, depth=3))
    train.transplant(base)
    wrapped, explicit = copy.deepcopy(base), copy.deepcopy(base)
    train.joint.install(wrapped, 'fixed_mixture')
    for layer in (explicit.conv3d_features[1], explicit.conv2d_features[1]):
        layer.momentum = .19
    source = torch.randn(2, 1, 48, 5, 5)
    target = torch.randn_like(source) * 2 + 3
    rng = torch.get_rng_state()
    xt = target.clone().requires_grad_(True)
    tokens = train.joint.paired_tokens(wrapped, source, xt)
    torch.set_rng_state(rng)
    cube = explicit.conv3d_features(torch.cat((source, target)))
    b, c, bands, h, w = cube.shape
    features = explicit.conv2d_features(cube.reshape(b, c * bands, h, w))
    fs, ft = features.split(2)
    expected = explicit._forward_semantic_tokens(fs), explicit._forward_semantic_tokens(ft)
    for a, b in zip(tokens, expected):
        torch.testing.assert_close(a, b)
    tokens[0].square().sum().backward()
    assert xt.grad is not None and xt.grad.abs().sum() > 0
    assert wrapped.spatial_queries.grad is not None and wrapped.spatial_queries.grad.abs().sum() > 0
    for name in ('conv3d_features.1', 'conv2d_features.1'):
        left, right = wrapped.get_submodule(name), explicit.get_submodule(name)
        torch.testing.assert_close(left.running_mean, right.running_mean)
        torch.testing.assert_close(left.running_var, right.running_var)
        assert int(left.num_batches_tracked) == 1
    wrapped.eval()
    restored = copy.deepcopy(base)
    train.joint.install(restored, 'fixed_mixture')
    restored.load_state_dict(wrapped.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(wrapped(source, source)[1], restored(source, source)[1])
    print('PASS: independent joint stem matches query tokens/buffers; target-moment and query gradients active; checkpoint reload exact')

if __name__ == '__main__':
    main()
