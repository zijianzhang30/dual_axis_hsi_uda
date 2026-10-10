"""Offline geometry, gated by all-seed frozen feature/prediction manifest."""
import csv
import json
import time
import numpy as np
from common import BASE, DATA, HERE, OUT, SEEDS, STAGES, dump, sha, ahash, unit

def writecsv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def stats(x):
    x = np.asarray(x, np.float64)
    return {'mean': float(x.mean()), 'std': float(x.std()), 'median': float(np.median(x)),
            'q05': float(np.quantile(x, .05)), 'q95': float(np.quantile(x, .95)),
            'negative_fraction': float((x < 0).mean()), 'positive_fraction': float((x > 0).mean())}

def centers(x, y, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    counts = np.bincount(y[mask], minlength=7)
    assert (counts > 1).all()
    means = np.stack([x[mask & (y == c)].mean(0, dtype=np.float64) for c in range(7)])
    assert (np.linalg.norm(means, axis=1) > 1e-12).all()
    return means, unit(means), counts

def loo_scores(x, y, means, counts, full_scores):
    # Correct-class sum excludes x; its scalar 1/(n-1) cancels under L2 normalization.
    scores = full_scores.copy()
    for c in range(7):
        ids = np.flatnonzero(y == c)
        for start in range(0, len(ids), 1024):
            ix = ids[start:start+1024]
            a = np.asarray(x[ix], np.float64)
            excluded = counts[c] * means[c] - a
            assert (np.linalg.norm(excluded, axis=1) > 1e-12).all()
            scores[ix, c] = np.einsum('nd,nd->n', a, unit(excluded))
    return scores

def margin(scores, y):
    wrong = scores.copy()
    wrong[np.arange(len(y)), y] = -np.inf
    return scores[np.arange(len(y)), y] - wrong.max(1)

def metric(y, pred):
    cm = np.bincount(7*y+pred, minlength=49).reshape(7, 7)
    recall = np.diag(cm)/cm.sum(1)
    return {'OA_percent': float(100*np.trace(cm)/len(y)), 'AA_percent': float(100*recall.mean()),
            'per_class_recall_percent': (100*recall).tolist(), 'confusion': cm.tolist()}

def geometry(source, target, sy, ty, train, logits):
    sm, sc, sn = centers(source, sy, train)
    tm, tc, tn = centers(target, ty)
    # Chunk full-map products to bound transient memory.
    ss = np.concatenate([np.asarray(target[i:i+1024], np.float64) @ sc.T for i in range(0, len(ty), 1024)])
    ts = np.concatenate([np.asarray(target[i:i+1024], np.float64) @ tc.T for i in range(0, len(ty), 1024)])
    ls = loo_scores(target, ty, tm, tn, ts)
    margins = {'target_nearest_class_margin': margin(ls, ty),
               'target_full_centroid_margin': margin(ts, ty),
               'target_to_source_correct_class_margin': margin(ss, ty)}
    if logits is not None:
        margins['native_logit_decision_margin'] = margin(logits.astype(np.float64), ty)
    srcpred, loopred = ss.argmax(1), ls.argmax(1)
    classes = []
    matrix = tc @ sc.T
    for c in range(7):
        mask = ty == c
        owncos = ts[mask, c]
        # mean||x-m||² = mean||x||² - ||m||² (m is arithmetic unit-feature mean).
        ids = np.flatnonzero(mask)
        totalnorm = 0.0
        for start in range(0, len(ids), 1024):
            a = np.asarray(target[ids[start:start+1024]], np.float64)
            totalnorm += float(np.einsum('nd,nd->', a, a))
        scatter = totalnorm / tn[c] - float(tm[c] @ tm[c])
        row = {'class': c+1, 'target_count': int(tn[c]), 'source_train_count': int(sn[c]),
               'target_intra_class_scatter': scatter, 'target_mean_cosine_distance_to_centroid': float((1-owncos).mean()),
               'source_centroid_target_recall_percent': float(100*(srcpred[mask] == c).mean()),
               'target_LOO_centroid_oracle_recall_percent': float(100*(loopred[mask] == c).mean()),
               'target_centroid_nearest_source_class': int(matrix[c].argmax()+1),
               'target_source_centroid_correct_cosine': float(matrix[c,c]),
               'target_source_centroid_correct_margin': float(matrix[c,c]-np.delete(matrix[c],c).max())}
        for key, values in margins.items():
            for k, v in stats(values[mask]).items():
                row[f'{key}_{k}'] = v
        classes.append(row)
    keys = [k for k in classes[0] if k not in ['class', 'target_count', 'source_train_count', 'target_centroid_nearest_source_class']]
    macro = {k: float(np.mean([r[k] for r in classes])) for k in keys}
    return {'dimension': source.shape[1], 'per_class': classes, 'macro': macro,
            'micro_margin_statistics': {k: stats(v) for k, v in margins.items()},
            'source_centroid_target_classification': metric(ty, srcpred),
            'target_LOO_centroid_oracle_classification': metric(ty, loopred),
            'source_target_centroid_cosine_rows_target_columns_source': matrix.tolist()}, {
                'source_centroids': sc, 'target_centroids': tc, 'source_cosine_scores': ss,
                'target_full_centroid_scores': ts, 'target_LOO_centroid_scores': ls,
                'source_centroid_predictions': srcpred, 'target_LOO_centroid_predictions': loopred, **margins}

def check_formulas():
    # Independent brute-force exclusion verifies LOO and scatter formulas on seven classes.
    rng = np.random.default_rng(2100)
    y = np.repeat(np.arange(7), 4)
    x = unit(rng.normal(size=(28, 9)))
    means, cent, counts = centers(x, y)
    scores = loo_scores(x, y, means, counts, x @ cent.T)
    for i in range(len(y)):
        correct = unit(x[(y == y[i]) & (np.arange(len(y)) != i)].mean(0))
        assert np.allclose(scores[i,y[i]], x[i] @ correct, atol=1e-12)
    for c in range(7):
        a = x[y == c]
        assert np.allclose(np.square(a-means[c]).sum(1).mean(), 1-means[c]@means[c], atol=1e-12)
    assert np.array_equal((scores.argmax(1) == y), margin(scores,y) > 0)

def main():
    start = time.monotonic()
    check_formulas()
    gate = json.loads((OUT / 'ALL_FEATURES_PREDICTIONS_FROZEN.json').read_text())
    assert gate['all_seeds_completed'] and not gate['target_GT_opened_by_extractor']
    assert gate['protocol_sha256'] == sha(HERE / 'PROTOCOL.txt')
    for path, h in gate['files_sha256'].items():
        assert sha(OUT / path) == h
    provenance = json.loads((OUT / 'extraction_provenance.json').read_text())
    for path, h in provenance['code_sha256'].items():
        assert sha(path) == h
    print('All frozen hashes verified. Opening offline labels now.', flush=True)
    sy = np.load(DATA / 'source_labels.npy')
    ty = np.load(DATA / 'target_labels.npy')
    train = np.load(DATA / 'source_probe_train_mask.npy')
    assert len(sy) == 2520 and len(ty) == 52901 and train.sum() == 2393
    metadata = provenance['historical_input_metadata']
    for domain, labels in [('source', sy), ('target', ty)]:
        assert ahash(labels) == metadata['domains'][domain]['labels_sha256']
        coords = np.load(OUT / 'seed_2100' / f'{domain}_coords.npy')
        gt = np.load(DATA / f'{domain}_gt.npy', mmap_mode='r')
        assert np.array_equal(labels, gt[coords[:,0],coords[:,1]])
    assert np.array_equal(ty, np.load(BASE / 'houston/evaluation_labels.npy'))
    runs, rows, macrorows, matrixrows, transitions = {}, [], [], [], []
    for seed in SEEDS:
        folder = OUT / f'seed_{seed}'
        logits = np.load(folder / 'target_logits.npy')
        predictions = np.load(folder / 'target_predictions.npy')
        assert np.array_equal(predictions, logits.argmax(1))
        runs[str(seed)] = {'native_classifier': metric(ty, predictions), 'stages': {}}
        for stage in STAGES:
            s = np.load(folder / f'source_{stage}.npy', mmap_mode='r')
            t = np.load(folder / f'target_{stage}.npy', mmap_mode='r')
            for a in [s, t]:
                for i in range(0, len(a), 1024):
                    norms = np.linalg.norm(a[i:i+1024], axis=1)
                    assert np.isfinite(a[i:i+1024]).all() and abs(norms-1).max() < 2e-6
            result, arrays = geometry(s, t, sy, ty, train, logits if stage == 'Z' else None)
            np.savez_compressed(folder / f'{stage}_offline_geometry.npz', **arrays)
            runs[str(seed)]['stages'][stage] = result
            rows.extend({'seed': seed, 'stage': stage, **r} for r in result['per_class'])
            macrorows.append({'seed': seed, 'stage': stage, **result['macro']})
            mat = np.asarray(result['source_target_centroid_cosine_rows_target_columns_source'])
            for c in range(7):
                for j in range(7):
                    matrixrows.append({'seed': seed, 'stage': stage, 'target_class': c+1, 'source_class': j+1, 'cosine': float(mat[c,j])})
            print(seed, stage, 'source-centroid AA', result['source_centroid_target_classification']['AA_percent'],
                  'target LOO AA', result['target_LOO_centroid_oracle_classification']['AA_percent'], flush=True)
        for a, b in [('V','T'), ('T','Z'), ('V','Z')]:
            ra, rb = runs[str(seed)]['stages'][a], runs[str(seed)]['stages'][b]
            for cls in range(8):
                x = ra['macro'] if cls == 7 else ra['per_class'][cls]
                y = rb['macro'] if cls == 7 else rb['per_class'][cls]
                keys = ['target_intra_class_scatter', 'target_nearest_class_margin_mean',
                        'target_to_source_correct_class_margin_mean', 'source_centroid_target_recall_percent',
                        'target_LOO_centroid_oracle_recall_percent', 'target_source_centroid_correct_margin']
                transitions.append({'seed': seed, 'transition': f'{a}->{b}', 'class': 'macro' if cls == 7 else cls+1,
                                    **{f'delta_{k}': y[k]-x[k] for k in keys}})
    # CSV union handles Z-only native logit fields without assigning fake values to V/T.
    def unioncsv(name, values):
        keys = list(dict.fromkeys(k for r in values for k in r))
        writecsv(OUT / name, [{k:r.get(k,'') for k in keys} for r in values])
    unioncsv('per_class.csv', rows)
    unioncsv('macro_per_seed.csv', macrorows)
    writecsv(OUT / 'centroid_cosine_matrices.csv', matrixrows)
    writecsv(OUT / 'stage_transitions.csv', transitions)
    aggregates, aggrows, aggclass = {}, [], []
    for stage in STAGES:
        rr = [runs[str(seed)]['stages'][stage] for seed in SEEDS]
        def aggregate(values):
            return {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1)), 'values': values}
        macroagg = {k: aggregate([r['macro'][k] for r in rr]) for k in rr[0]['macro']}
        matrices = np.asarray([r['source_target_centroid_cosine_rows_target_columns_source'] for r in rr])
        aggregates[stage] = {'macro': macroagg, 'centroid_cosine_mean': matrices.mean(0).tolist(),
                             'centroid_cosine_sample_std': matrices.std(0,ddof=1).tolist()}
        for k, v in macroagg.items():
            aggrows.append({'stage': stage, 'metric': k, **v})
        for c in range(7):
            for k in macroagg:
                aggclass.append({'stage': stage, 'class': c+1, 'metric': k, **aggregate([r['per_class'][c][k] for r in rr])})
    writecsv(OUT / 'macro_seed_aggregate.csv', aggrows)
    writecsv(OUT / 'per_class_seed_aggregate.csv', aggclass)
    limitations = ['Frozen epoch200 C only, no trained probe, no checkpoint or hyperparameter selection.',
                  'Target LOO centroids are offline label-assisted diagnostics, not deployable predictions.',
                  'Different descriptor dimensions and pooling make stage changes descriptive, not causal.',
                  'LOO removes self-inclusion but not spatial dependence of overlapping patches.',
                  'Three seeds on one scene pair do not establish statistical significance; target C4 has22 pixels.']
    dump(OUT / 'summary.json', {'runs': runs, 'aggregate': aggregates, 'stage_transitions': transitions,
         'limitations': limitations, 'formulas_bruteforce_verified': True, 'seconds': time.monotonic()-start})
    report(runs, aggregates, transitions, limitations)
    for path, h in gate['files_sha256'].items():
        assert sha(OUT / path) == h
    for rec in provenance['seeds'].values():
        assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    dump(OUT / 'SCORING_COMPLETE.json', {'frozen_artifacts_and_checkpoints_unchanged': True,
         'labels_first_opened_after_all_seed_freeze_and_hash_verification': True,
         'no_training_no_followup': True, 'label_file_sha256': {str(DATA/f'{d}_labels.npy'): sha(DATA/f'{d}_labels.npy') for d in ['source','target']},
         'report_sha256': sha(OUT/'REPORT.txt'), 'summary_sha256': sha(OUT/'summary.json')})

def report(runs, aggregates, transitions, limitations):
    lines = ['Houston class geometry diagnostic V1; Joint BN+C; original seeds2100/2101/2102; fixed epoch200.',
             'Features and predictions of all seeds frozen before target GT access. No training or follow-up.',
             'V=full CNN map flattened10816; T=semantic tokens flattened256; Z=CLS64. All sample features L2 normalized.',
             'Source centroids use2393 original train samples; target all52901. Class macro is unweighted.',
             'Scatter=mean squared distance to arithmetic mean. Main target margin is leave-one-out.',
             'Matrices: rows target classes1..7, columns source classes1..7. Native logits remain raw.', '',
             'Three-seed macro mean +/- sampleSD:']
    keys = ['target_intra_class_scatter', 'target_nearest_class_margin_mean', 'target_to_source_correct_class_margin_mean',
            'source_centroid_target_recall_percent', 'target_LOO_centroid_oracle_recall_percent']
    for stage in STAGES:
        lines.append(stage)
        for key in keys:
            v=aggregates[stage]['macro'][key]
            lines.append(f'  {key}: {v["mean"]:.6f} +/- {v["sample_std"]:.6f}')
        lines.append('  Mean7x7 target-source centroid cosine:')
        lines.extend('  '+' '.join(f'{v:.5f}' for v in row) for row in aggregates[stage]['centroid_cosine_mean'])
    lines.append('\nNative target classifier:')
    for seed in SEEDS:
        r=runs[str(seed)]['native_classifier']
        lines.append(f'  seed{seed}: OA={r["OA_percent"]:.5f}, AA={r["AA_percent"]:.5f}')
    v=aggregates['Z']['macro']['native_logit_decision_margin_mean']
    lines.append(f'Z macro true-vs-strongest-wrong native logit margin: {v["mean"]:.6f} +/- {v["sample_std"]:.6f}')
    lines.append('\nPaired stage changes (positive margin/recall better; negative scatter better):')
    for transition in ['V->T','T->Z','V->Z']:
        rr=[r for r in transitions if r['transition']==transition and r['class']=='macro']
        lines.append(transition)
        for key in [k for k in rr[0] if k.startswith('delta_')]:
            values=np.array([r[key] for r in rr]);lines.append(f'  {key}: {values.mean():+.6f} +/- {values.std(ddof=1):.6f}; seeds={values.tolist()}')
    lines.append('\nPer-class stage correspondence (macro across seeds):')
    for c in range(7):
        lines.append(f'C{c+1}')
        for stage in STAGES:
            rr=[runs[str(s)]['stages'][stage]['per_class'][c] for s in SEEDS]
            lines.append(f'  {stage}: scatter={np.mean([r["target_intra_class_scatter"] for r in rr]):.5f}; '
                         f'targetLOO margin={np.mean([r["target_nearest_class_margin_mean"] for r in rr]):+.5f}; '
                         f'source margin={np.mean([r["target_to_source_correct_class_margin_mean"] for r in rr]):+.5f}; '
                         f'source-centroid recall={np.mean([r["source_centroid_target_recall_percent"] for r in rr]):.2f}%; '
                         f'targetLOO recall={np.mean([r["target_LOO_centroid_oracle_recall_percent"] for r in rr]):.2f}%; '
                         f'nearest source centroid={[r["target_centroid_nearest_source_class"] for r in rr]}')
    lines.extend(['\nLimitations:', *limitations])
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,stage in zip(axes,STAGES):
        mat=np.array(aggregates[stage]['centroid_cosine_mean'])
        im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set(title=f'{stage}: mean of 3 seeds',xlabel='Source class',ylabel='Target class',xticks=range(7),yticks=range(7),xticklabels=range(1,8),yticklabels=range(1,8))
        for i in range(7):
            for j in range(7):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axes,label='Centroid cosine')
    fig.savefig(OUT/'centroid_cosine_matrices.png',dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
