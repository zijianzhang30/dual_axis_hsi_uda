"""Two additional joint-BN arms with the same released BiDA objective."""
from pathlib import Path
import difflib

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_joint_adapt_2x2_v1')
BASE = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1/derived_main.py')
source = BASE.read_text()
derived = source.replace("choices=['A']", "choices=['B_adapt', 'C_adapt']")
derived = derived.replace("results/tstnet_pipeline_bida_v1')", "results/tstnet_joint_adapt_2x2_v1')")
derived = derived.replace("PROJECT / 'experiments/bn_stabilization_v1/normalization.py'", "ROOT / 'frozen_components/normalization.py'")
derived = derived.replace("PROJECT / 'experiments/relation_tokenizer_v1/relation.py'", "ROOT / 'frozen_components/relation.py'")
old = "        ema = get_model('BiDA', 'Dioni', patch_size, opts, ema=True).cuda()"
new = """        ema = get_model('BiDA', 'Dioni', patch_size, opts, ema=True)
        joint.install(model, 'fixed_mixture')
        joint.install(ema, 'fixed_mixture')
        if own.arm == 'C_adapt':
            relation.install(model, 'C', 1233)
            relation.install(ema, 'C', 1234)
        for parameter in ema.parameters():
            parameter.detach_()
        assert [name for name, _ in model.named_parameters()] == [name for name, _ in ema.named_parameters()]
        ema = ema.cuda()"""
assert derived.count(old) == 1
derived = derived.replace(old, new)
derived = derived.replace("'bn_momentum': .1,", "'bn_momentum': .19,")
derived = derived.replace("'ema_buffers': 'native teacher own BN buffers, no buffer EMA'", "'ema_buffers': 'own Full Joint BN buffers, no buffer EMA', 'teacher_relation_seed': 1234 if own.arm == 'C_adapt' else None")
needle = "        checks = []\n        for epoch in [100, 101]:"
replacement = """        small_source = torch.randn(2, 1, 176, 12, 12, device='cuda')
        small_target = torch.randn_like(small_source).requires_grad_(True)
        ce = F.cross_entropy(model(small_source, small_target)[0], torch.tensor([0, 1], device='cuda'))
        target_gradient = torch.autograd.grad(ce, small_target)[0]
        assert torch.isfinite(target_gradient).all() and target_gradient.abs().sum() > 0
        checks = [dict(probe='CE target gradient through joint moments', gradient_l1=float(target_gradient.abs().sum()))]
        for epoch in [100, 101]:"""
assert derived.count(needle) == 1
derived = derived.replace(needle, replacement)
needle = "            (outdir / 'config.json').write_text(json.dumps(config, indent=2))"
replacement = """            ce_arm = 'B' if own.arm == 'B_adapt' else 'C'
            old = json.loads((Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1') / ce_arm / 'config.json').read_text())
            matched_keys = ['seed', 'epochs', 'patch', 'batch', 'source_fraction', 'source_repeat', 'source_split', 'lr', 'momentum', 'weight_decay', 'lr_schedule', 'optimizer_recreated_each_epoch', 'native_grid', 'source_count', 'target_train_count', 'target_eval_count', 'source_steps_per_epoch', 'index_hashes', 'labels_hashes', 'shared_initial_sha256', 'initial_sha256', 'bn_momentum', 'parameters']
            config['CE_pair_config_checks'] = {key: config[key] == old[key] for key in matched_keys}
            if not all(config['CE_pair_config_checks'].values()):
                (outdir / 'mismatch.json').write_text(json.dumps(config, indent=2))
                raise RuntimeError('Mismatch with frozen CE control before training')
            (outdir / 'config.json').write_text(json.dumps(config, indent=2))"""
assert derived.count(needle) == 1
derived = derived.replace(needle, replacement)
ROOT.mkdir(parents=True, exist_ok=True)
for name, content in [('derived_main.py', derived), ('adapter.patch', ''.join(difflib.unified_diff(source.splitlines(True), derived.splitlines(True), fromfile='full_BiDA_TSTnet_pipeline', tofile='fixed_joint_BN_adaptation_arms')))]:
    path = ROOT / name
    if path.exists():
        assert path.read_text() == content
    else:
        path.write_text(content)
exec(compile(derived, str(ROOT / 'derived_main.py'), 'exec'), {'__name__': '__main__', '__file__': str(ROOT / 'derived_main.py')})
