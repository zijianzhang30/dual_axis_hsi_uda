"""Check initialization truncation and non-invasive layer hooks."""
import copy
from types import SimpleNamespace
import torch
import train
from layers import LayerCapture

def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    base = train.get_model('BiDA', 'Houston13', 13, SimpleNamespace(num_tokens=4, dim=64, depth=3))
    train.transplant(base)
    shallow = copy.deepcopy(base)
    shallow.blocks = torch.nn.ModuleList([shallow.blocks[0]])
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in shallow.state_dict().items())
    for model in (base, shallow):
        train.joint.install(model, 'fixed_mixture')
        model.eval()
        x = torch.randn(2, 1, 48, 5, 5)
        with torch.no_grad():
            before = model(x, x)[1]
            capture = LayerCapture(model)
            after = model(x, x)[1]
            assert torch.equal(before, after)
            for i in range(1, len(model.blocks)+1):
                v = capture.values
                torch.testing.assert_close(v[f'block{i}_after_attention'], v[f'block{i}_input_cls'] + v[f'block{i}_attention_update'])
                torch.testing.assert_close(v[f'block{i}_after_mlp'], v[f'block{i}_after_attention'] + v[f'block{i}_mlp_update'])
            capture.close()
            model.train()
            xt = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x, xt)[0].square().sum().backward()
        assert xt.grad is not None and xt.grad.abs().sum() > 0
    print('PASS: retained initialization exact; hooks preserve logits; attention/MLP residual boundaries correct; Full Joint target gradients active at both depths')

if __name__ == '__main__':
    main()
