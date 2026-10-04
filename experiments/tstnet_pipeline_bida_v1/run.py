"""Full released BiDA optimization under the already frozen TSTnet protocol."""
from pathlib import Path
import difflib

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1')
BASE = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1/source_snapshot/run.py')
source = BASE.read_text()
derived = source.replace("choices=['B', 'C']", "choices=['A']")
derived = derived.replace("results/tstnet_pipeline_self_v1')", "results/tstnet_pipeline_bida_v1')")
derived = derived.replace("'/home/zhangzj26/IEEE_TCSVT_BiDA'", "str(ROOT / 'bida_snapshot')")
start = derived.index('    def build(bands, num_classes, patch_size):')
stop = derived.index('    if own.preflight:', start)
replacement = '''    import train_pipeline as released
    from loss import make_loss
    opts = SimpleNamespace(dim=64, depth=3, num_tokens=4, bs=100,
                           lambda1=.1, lambda2=1., ema_decay=.999,
                           loss_type='softmax', labelsmooth='off', device='0')
    native_tree = ast.parse((ROOT / 'bida_snapshot/train_pipeline.py').read_text())
    fn = next(n for n in native_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    outer = next(n for n in fn.body if isinstance(n, ast.For))
    inner = next(n for n in outer.body if isinstance(n, ast.For))
    batch_code = compile(ast.Module(body=inner.body, type_ignores=[]), str(ROOT / 'bida_snapshot/train_pipeline.py'), 'exec')
    batch_ns = dict(released.__dict__)
    batch_ns.update(opts=opts, device=torch.device('cuda:0'), global_step=0,
                    losses=[], Align_losses=[], Distill_losses=[],
                    MMD_criterion=released.MMD_loss())

    def build(bands, num_classes, patch_size):
        assert bands == 176 and num_classes == 12 and patch_size == 12
        model = get_model('BiDA', 'Dioni', patch_size, opts)
        shared_initial = state_hash(model)
        ema = get_model('BiDA', 'Dioni', patch_size, opts, ema=True).cuda()
        criterion, center = make_loss(opts, 12)
        batch_ns.update(network=model, network_ema=ema, criterion=criterion)
        ns['_ema'] = ema
        ns['_unused_center'] = center
        ns['_model_audit'] = {'shared_initial_sha256': shared_initial,
            'parameters': sum(x.numel() for x in model.parameters()), 'bn_momentum': .1,
            'initial_sha256': state_hash(model), 'ema_initial_sha256': state_hash(ema),
            'lambda1': .1, 'lambda2': 1., 'ema_decay': .999,
            'loss_body': 'unmodified released AST inner batch body',
            'mmd_enable_condition': 'epoch > 100 AND full batch100',
            'ema_buffers': 'native teacher own BN buffers, no buffer EMA'}
        return model

'''
derived = derived[:start] + replacement + derived[stop:]
start = derived.index('    if own.preflight:')
stop = derived.index("    if (outdir / 'history.jsonl').exists():", start)
replacement = '''    if own.preflight:
        ns['seed_worker'](1233)
        model = build(176, 12, 12).cuda().train()
        checks = []
        for epoch in [100, 101]:
            source = torch.randn(100, 1, 176, 12, 12, device='cuda')
            target = torch.randn_like(source)
            labels = torch.arange(100, device='cuda') % 12
            optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=1e-4)
            batch_ns.update(data_src=(source, labels), data_tar=(target, torch.zeros_like(labels)),
                            e=epoch, optimizer=optimizer, network=model)
            previous = len(batch_ns['Align_losses'])
            exec(batch_code, batch_ns)
            loss = batch_ns['loss']
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            enabled = len(batch_ns['Align_losses']) > previous
            assert enabled == (epoch == 101)
            checks.append(dict(epoch=epoch, loss=float(loss), mmd_executed=enabled,
                               global_step=batch_ns['global_step']))
        model.eval()
        with torch.no_grad():
            assert model(source[:2], source[:2])[1].shape == (2, 12)
        audit = dict(ns['_model_audit'], checks=checks)
        (outdir / 'preflight.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit), flush=True)
        return

'''
derived = derived[:start] + replacement + derived[stop:]
derived = derived.replace("objective='source CE only'", "objective='complete released BiDA CE/distillation/EMA/MMD'")
old = '''            optimizer.zero_grad()
            logits = model(source.unsqueeze(1), target.unsqueeze(1))[0]
            loss = F.nll_loss(F.log_softmax(logits, dim=1), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; no automatic correction')
            loss.backward()
            optimizer.step()
'''
new = '''            batch_ns.update(data_src=(source.unsqueeze(1), labels),
                            data_tar=(target.unsqueeze(1), torch.zeros_like(labels)),
                            e=epoch + 1, optimizer=optimizer, network=model)
            exec(batch_code, batch_ns)
            logits, loss = batch_ns['out_x'], batch_ns['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; no automatic correction')
'''
assert derived.count(old) == 1
derived = derived.replace(old, new)
derived = derived.replace('        ns[\'_epoch\'] = epoch + 1',
    "        record.update(global_step=batch_ns['global_step'], mmd_batches=len(batch_ns['Align_losses']), source_ce=float(batch_ns['loss_cls']), total_loss=record.pop('source_ce'))\n        batch_ns['losses'].clear(); batch_ns['Align_losses'].clear(); batch_ns['Distill_losses'].clear()\n        ns['_epoch'] = epoch + 1")
derived = derived.replace("            torch.save(model.state_dict(), outdir / 'final_epoch500.pth')",
    "            torch.save(model.state_dict(), outdir / 'final_epoch500.pth')\n            torch.save(ns['_ema'].state_dict(), outdir / 'ema_epoch500.pth')")
ROOT.mkdir(parents=True, exist_ok=True)
for name, text in [('derived_main.py', derived), ('adapter.patch', ''.join(difflib.unified_diff(source.splitlines(True), derived.splitlines(True), fromfile='frozen_BC_pipeline', tofile='full_BiDA_pipeline')))]:
    path = ROOT / name
    if path.exists():
        assert path.read_text() == text, 'Frozen adapter changed'
    else:
        path.write_text(text)
exec(compile(derived, str(ROOT / 'derived_main.py'), 'exec'), {'__name__': '__main__', '__file__': str(ROOT / 'derived_main.py')})
