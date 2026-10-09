"""Audit and export existing frozen results without mutating any old artifact."""
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path('/home/zhangzj26/dual_axis_hsi_uda')
OLD = ROOT / 'results/official_bench_v1'
OUT = Path('/nas1/zhangzj26/dual_axis_hsi_uda/results/cross_benchmark_frozen_v1')
SEEDS = (2100, 2101, 2102)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export_csv(name, rows):
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    assert not OUT.exists(), 'Refuse overwrite'
    old = json.loads((OLD / 'summary.json').read_text())
    upstream = json.loads((OLD / 'audit_A_v1/protocol_checks.json').read_text())
    manifests, histories, audit = {}, {}, []
    for seed in SEEDS:
        for arm in ('A', 'C'):
            folder = OLD / ('hyrank_%s_%s' % (arm, seed))
            m = json.loads((folder / 'frozen_manifest.json').read_text())
            cfg = m['config']
            h = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
            assert len(h) == 200 and [r['epoch'] for r in h] == list(range(1, 201))
            assert all(r['steps'] == 80 for r in h)
            assert cfg['settings']['epoch'] == 200 and cfg['settings']['patch_size'] == 13
            assert cfg['settings']['bs'] == 128 and cfg['seed'] == seed
            for path, expected in cfg['code_sha256'].items():
                assert sha(path) == expected, path
            assert sha(m['prediction']['path']) == m['prediction']['sha256']
            final = m['checkpoints'][-1]
            assert final['epoch'] == 200 and sha(final['path']) == final['sha256']
            if arm == 'C':
                assert cfg['objective'] == 'source-self CE only' and cfg['bn'] == 'Full Joint'
            manifests[arm, seed], histories[arm, seed] = m, h
        a, c = [manifests[arm, seed]['config'] for arm in ('A', 'C')]
        assert a['data_protocol'] == c['data_protocol']
        assert a['initial_common_model_sha256'] == c['initial_common_model_sha256']
        assert a['settings'] == c['settings'] or {
            k:v for k,v in a['settings'].items() if k != 'device'
        } == {k:v for k,v in c['settings'].items() if k != 'device'}
        assert [r['batch_stream_sha256'] for r in histories['A', seed]] == [
            r['batch_stream_sha256'] for r in histories['C', seed]]
        audit.append({'seed':seed, 'data_indices_equal':True, 'common_initialization_equal':True,
                      'batch_augmentation_stream_equal_all_200_epochs':True,
                      'endpoint_checkpoint_and_prediction_hash_verified':True,
                      'steps_per_epoch':80, 'optimizer_steps':16000})
    for name in ('main.py', 'train_pipeline.py', 'models/BiDA.py', 'models/get_model.py',
                 'utils/scheduler.py', 'loss/make_loss.py', 'loss/mmd_loss.py'):
        p = OLD / 'audit_A_v1/upstream' / name
        local = Path('/home/zhangzj26/IEEE_TCSVT_BiDA') / name
        assert ast.dump(ast.parse(p.read_text())) == ast.dump(ast.parse(local.read_text()))
    rows, perclass, pairs = [], [], []
    for seed in SEEDS:
        for arm in ('A', 'C'):
            r = next(r for r in old['rows'] if r['arm'] == arm and r['seed'] == seed)
            cm = np.asarray(r['confusion_matrix'])
            oa = float(np.trace(cm) / cm.sum() * 100)
            recall = np.diag(cm) / cm.sum(1) * 100
            assert abs(oa-r['oa_percent']) < 1e-10
            assert np.max(np.abs(recall-r['per_class_percent'])) < 1e-10
            zero = [int(i+1) for i in np.flatnonzero(recall == 0)]
            rows.append({'benchmark':'HyRANK Dioni->Loukia','model':'released BiDA' if arm=='A' else 'Ours',
                         'seed':seed,'epoch':200,'OA':oa,'AA':float(recall.mean()),'Kappa':r['kappa'],
                         'collapse_legacy_any_zero_or_share_ge_95':r['collapse'],
                         'prediction_concentration_ge_95':max(r['prediction_shares']) >= .95,
                         'zero_recall_classes':';'.join(map(str,zero)),
                         'parameters':r['parameters'],'train_seconds':r['train_seconds']})
            for k, value in enumerate(recall):
                perclass.append({'benchmark':'HyRANK','model':rows[-1]['model'],'seed':seed,
                                 'class':k+1,'n_target':int(cm[k].sum()),'recall_percent':float(value),
                                 'prediction_share':r['prediction_shares'][k]})
        a, c = rows[-2:]
        pairs.append({'seed':seed,'delta_OA_pp':c['OA']-a['OA'],
                      'delta_AA_pp':c['AA']-a['AA'],'delta_Kappa':c['Kappa']-a['Kappa']})
    aggregate = {}
    for model in ('released BiDA','Ours'):
        rr = [r for r in rows if r['model']==model]
        aggregate[model] = {key:{'mean':float(np.mean([r[key] for r in rr])),
                                'sample_std':float(np.std([r[key] for r in rr],ddof=1))}
                            for key in ('OA','AA','Kappa')}
    paired = {key:{'mean':float(np.mean([r[key] for r in pairs])),
                  'sample_std':float(np.std([r[key] for r in pairs],ddof=1))}
              for key in ('delta_OA_pp','delta_AA_pp','delta_Kappa')}
    paper_rows = []
    for benchmark, table, oa, kappa, recalls in (
        ('HyRANK Dioni->Loukia','XIV',68.89,.627,[34.15,80.19,63.62,40.55,69.20,36.02,81.22,57.52,50.38,50.68,100,100]),
        ('MFF-SD->TD1','XI',77.40,.7176,[97.69,95.26,75.11,44.35,74.46]),
        ('MFF-SD->TD2','XII',75.08,.6725,[90.76,81.11,85.02,24.71,60.23])):
        paper_rows.append({'benchmark':benchmark,'reported_OA':oa,'reported_Kappa':kappa,
                           'AA_derived_from_rounded_class_recalls':float(np.mean(recalls)),
                           'table':table,'source':'https://arxiv.org/html/2507.02268v1',
                           'version':'author preprint v1; IEEE final numerical identity not verified',
                           'seed_and_checkpoint_rule':'not established; never merged with local results'})
    caveats = [
        'Existing matched HyRANK endpoint reused; no new training and no changed architecture.',
        'Full BiDA released optimization loop retained; target-best block replaced with common target-blind fixed-endpoint/source-val callback.',
        'Public HSIDataset has self.data missing; existing local runtime assignment repaired it. Native strict boundary filtering retained.',
        'Source validation epoch1/every10 consumes shared DataLoader generator: paired A/C matches, but not bit-identical to untouched release target-evaluation cadence.',
        'Ours trains only source-self CE; unused native coupled outputs are still computed to preserve frozen execution/RNG, with no coupled/distillation/EMA/MMD optimization.',
        'Official generic defaults used; undocumented benchmark-specific overrides not invented.',
        'Legacy collapse includes ANY zero recall; report prediction concentration and zero recall separately.',
        'Target candidate centers come from nonzero author GT mask; label values used only offline, never in losses/selection.',
        'MFF original SD/TD1/TD2 data and official loader/splits unavailable; neither MFF task launched.',
        'Published numbers are author-preprint paper numbers, not release reproductions or formal fixed-endpoint comparisons.'
    ]
    OUT.mkdir()
    export_csv('hyrank_fixed200.csv',rows)
    export_csv('hyrank_per_class_recall.csv',perclass)
    export_csv('hyrank_paired_deltas.csv',pairs)
    export_csv('paper_reported_separate.csv',paper_rows)
    result = {'status':'HyRANK completed/reused after audit; MFF two tasks blocked',
              'frozen_seeds':SEEDS,'upstream_commit':upstream['upstream_commit'],
              'audit':audit,'aggregate':aggregate,'paired_aggregate':paired,
              'paper_numbers_separate':paper_rows,'caveats':caveats,
              'historical_result_directory':str(OLD.resolve())}
    (OUT/'audit_and_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['Frozen cross-benchmark validation, 2026-10-07',
             'HyRANK formal endpoint: epoch200, seeds2100/2101/2102. No new model training.',
             'MFF-SD->TD1 / TD2: BLOCKED, need author data and official loader/split.', '']
    for model, values in aggregate.items():
        lines.append(model+': '+', '.join('%s %.4f +/- %.4f' % (k,v['mean'],v['sample_std']) for k,v in values.items()))
    for r in pairs:lines.append('paired '+json.dumps(r))
    lines += ['', 'Conclusion: HyRANK OA mean decreases; only seed2101 improves both OA/AA. No stable superiority.',
              'Paper BiDA OA (SEPARATE author-preprint table): HyRANK68.89, MFF-TD1 77.40, MFF-TD2 75.08.',
              '', 'Protocol disclosures:'] + caveats
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
