"""Derive two corrected C tasks without modifying any running dependency."""
import argparse
import difflib
from pathlib import Path
import sys

ROOT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_anchor_fixed_v1')
parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=['C', 'C_adapt'], required=True)
parser.add_argument('--preflight', action='store_true')
args = parser.parse_args()
if args.arm == 'C':
    base = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1/source_snapshot/run.py')
    old_root = 'tstnet_pipeline_self_v1'
    source = base.read_text()
    derived = source.replace("choices=['B', 'C']", "choices=['C']")
    derived = derived.replace("'/home/zhangzj26/IEEE_TCSVT_BiDA'", "str(ROOT / 'bida_snapshot')")
    derived = derived.replace("PROJECT / 'experiments/bn_stabilization_v1/normalization.py'", "ROOT / 'frozen_components/normalization.py'")
    derived = derived.replace("PROJECT / 'experiments/relation_tokenizer_v1/relation.py'", "ROOT / 'frozen_components/relation.py'")
else:
    base = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_joint_adapt_2x2_v1/derived_main.py')
    old_root = 'tstnet_joint_adapt_2x2_v1'
    source = base.read_text()
    derived = source.replace("choices=['B_adapt', 'C_adapt']", "choices=['C_adapt']")
derived = derived.replace("results/" + old_root + "')", "results/tstnet_anchor_fixed_v1')")
derived = derived.replace("    args = ns['args']", "    args = ns['args']\n    import anchors\n    ns['HyperX'], ns['data_prefetcher'] = anchors.tracked_loader_classes()")
for name, seed in [('model', 1233), ('ema', 1234)]:
    old = "            relation.install(%s, 'C', %s)" % (name, seed)
    if old in derived:
        assert derived.count(old) == 1
        derived = derived.replace(old, old + "\n            anchors.install_model(%s, allow_default=own.preflight)" % name)
derived = derived.replace('            source, labels = source_iter.next()', '            source, labels, source_anchor = source_iter.next()')
derived = derived.replace('            target, _ = target_iter.next()', '            target, _, target_anchor = target_iter.next()')
derived = derived.replace('            labels = (labels - 1).long()', "            labels = (labels - 1).long()\n            anchors.set_context(model, source_anchor, target_anchor)\n            if '_ema' in ns:\n                anchors.set_context(ns['_ema'], source_anchor, target_anchor)\n            coordinates = torch.cat((source_anchor, target_anchor), dim=0).detach().cpu().numpy()\n            for row, col in coordinates:\n                anchor_counts[(int(row), int(col))] = anchor_counts.get((int(row), int(col)), 0) + 1")
derived = derived.replace('        label_hash = hashlib.sha256()', '        label_hash = hashlib.sha256()\n        anchor_counts = {}')
derived = derived.replace("        ns['_epoch'] = epoch + 1", "        record['actual_anchor_counts'] = {str(key): value for key, value in anchor_counts.items()}\n        baseline_root = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/%s')\n        baseline_history = baseline_root / own.arm / 'history.jsonl'\n        old_records = [json.loads(x) for x in baseline_history.read_text().splitlines()]\n        match = next((r for r in old_records if r['epoch'] == epoch + 1), None)\n        if match is not None:\n            assert record['actual_consumed_source_labels_sha256'] == match['actual_consumed_source_labels_sha256'], 'Source batch trajectory changed'\n            record['source_batch_trace_matches_original'] = True\n        ns['_epoch'] = epoch + 1" % old_root)
derived = derived.replace("            for image, label in ns['test_loader']:", "            for image, label, coordinate in ns['test_loader']:")
derived = derived.replace("                image = image.cuda().unsqueeze(1)", "                image = image.cuda().unsqueeze(1)\n                coordinate = coordinate.cuda()\n                anchors.set_context(model, coordinate, coordinate)")
needle = "            (outdir / 'config.json').write_text(json.dumps(config, indent=2))"
replacement = """            old_config = json.loads((Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/OLD_ROOT') / own.arm / 'config.json').read_text())
            keys = ['seed', 'epochs', 'patch', 'batch', 'source_fraction', 'source_repeat', 'source_split', 'lr', 'momentum', 'weight_decay', 'lr_schedule', 'optimizer_recreated_each_epoch', 'native_grid', 'source_count', 'target_train_count', 'target_eval_count', 'source_steps_per_epoch', 'index_hashes', 'labels_hashes', 'shared_initial_sha256', 'initial_sha256', 'bn_momentum', 'parameters']
            config['original_pair_checks'] = {key: config[key] == old_config[key] for key in keys}
            assert all(config['original_pair_checks'].values()), 'Frozen config mismatch before first update'
            config['anchor_fix'] = 'marker follows actual native flip; no additional random draws; no label changes'
            (outdir / 'config.json').write_text(json.dumps(config, indent=2))""".replace('OLD_ROOT', old_root)
assert derived.count(needle) == 1
derived = derived.replace(needle, replacement)
assert 'source, labels, source_anchor = source_iter.next()' in derived
assert 'anchors.install_model(model' in derived
assert args.arm == 'C' or 'anchors.install_model(ema' in derived
folder = ROOT / args.arm
folder.mkdir(parents=True, exist_ok=True)
for name, text in [('derived_main.py', derived), ('anchor_fix.patch', ''.join(difflib.unified_diff(source.splitlines(True), derived.splitlines(True), fromfile='frozen_original', tofile='anchor_fixed')))]:
    path = folder / name
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
sys.argv = ['derived_main.py', '--arm', args.arm] + (['--preflight'] if args.preflight else [])
exec(compile(derived, str(folder / 'derived_main.py'), 'exec'), {'__name__': '__main__', '__file__': str(folder / 'derived_main.py')})
