"""Score only after all nine fixed-200 predictions have been frozen."""
import json
import numpy as np
import common as c


def main():
    records={}; histories={}
    for seed in (2100,2101,2102):
        for arm in 'ABC':
            out=c.OUT/f'hyrank_{arm}_{seed}'
            manifest=json.loads((out/'frozen_manifest.json').read_text())
            pred=manifest['prediction']
            assert c.helper.file_hash(pred['path'])==pred['sha256']
            assert [r['epoch'] for r in manifest['checkpoints']]==list(range(10,201,10))
            records[arm,seed]=manifest
            histories[arm,seed]=[json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
            assert len(histories[arm,seed])==200
        base=records['A',seed]['config']
        for arm in 'BC':
            cfg=records[arm,seed]['config']
            assert cfg['data_protocol']==base['data_protocol']
            assert cfg['initial_common_model_sha256']==base['initial_common_model_sha256']
            assert [r['batch_stream_sha256'] for r in histories[arm,seed]]==[r['batch_stream_sha256'] for r in histories['A',seed]]
        assert [r['rng_stream_sha256'] for r in histories['B',seed]]==[r['rng_stream_sha256'] for r in histories['C',seed]]
    # The barrier above verifies every prediction before accessing label values for scoring.
    loaders,_=c.loaders(c.options())
    labels=np.asarray(loaders[-1].dataset.labels)
    rows=[]
    for (arm,seed),manifest in records.items():
        with np.load(manifest['prediction']['path']) as pred:
            metrics=c.helper.metrics(labels,pred['predictions'],pred['logits'],12)
        row={'arm':arm,'seed':seed,**metrics,
             'collapse':bool(max(metrics['prediction_shares'])>=.95 or min(metrics['per_class_percent'])==0),
             'source_val_oa':manifest['source_val_at_final']['oa_percent'],
             'parameters':manifest['config']['parameters'],
             'train_seconds':manifest['train_seconds'],'inference_seconds':manifest['inference_seconds']}
        rows.append(row)
        (c.OUT/f'hyrank_{arm}_{seed}'/'results.json').write_text(json.dumps(row,indent=2)+'\n')
    fields=('oa_percent','aa_percent','kappa','source_val_oa','train_seconds')
    aggregate={arm:{field:{'mean':float(np.mean([r[field] for r in rows if r['arm']==arm])),
                          'sample_std':float(np.std([r[field] for r in rows if r['arm']==arm],ddof=1))}
                        for field in fields} for arm in 'ABC'}
    paired={}
    for other in 'BA':
        values=[]
        for seed in (2100,2101,2102):
            get=lambda arm:next(r for r in rows if r['arm']==arm and r['seed']==seed)
            values.append({'seed':seed,**{field:get('C')[field]-get(other)[field] for field in fields}})
        paired[f'C-{other}']=values
    summary={'endpoint':'fixed epoch 200; no target-selected model','rows':rows,
             'aggregate':aggregate,'paired':paired,
             'collapse_definition':'any zero-recall class OR max predicted-class share >=95%',
             'paper_comparison':'Not interchangeable with paper target-selected or unspecified endpoints.'}
    (c.OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# HyRANK official benchmark validation','',
           'Dioni → Loukia, author out68 masks, fixed epoch 200. Mean ± sample SD over seeds 2100/2101/2102.',
           'A: released full BiDA; B: joint-BN self/original tokenizer; C: joint-BN self/center-relative tokenizer.',
           'All prediction files were frozen before scoring; batch streams and B/C RNG streams matched.',
           '', '| Arm | OA (%) | AA (%) | Kappa | Source-val OA (%) | Collapse | Parameters | Training seconds |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for arm in 'ABC':
        fmt=lambda f:f"{aggregate[arm][f]['mean']:.3f} ± {aggregate[arm][f]['sample_std']:.3f}"
        rr=[r for r in rows if r['arm']==arm]
        lines.append(f"| {arm} | {fmt('oa_percent')} | {fmt('aa_percent')} | {fmt('kappa')} | {fmt('source_val_oa')} | {sum(r['collapse'] for r in rr)}/3 | {rr[0]['parameters']} | {fmt('train_seconds')} |")
    lines+=['','## Per-seed metrics and per-class recall','',
            '| Arm/seed | OA | AA | Kappa | Recall classes 1–12 (%) | Prediction shares classes 1–12 |',
            '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        recall=', '.join(f'{v:.2f}' for v in r['per_class_percent'])
        shares=', '.join(f'{v:.4f}' for v in r['prediction_shares'])
        lines.append(f"| {r['arm']}/{r['seed']} | {r['oa_percent']:.3f} | {r['aa_percent']:.3f} | {r['kappa']:.4f} | {recall} | {shares} |")
    lines+=['','## Paired fixed-200 deltas','', '| Comparison / seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |','| --- | --- | --- | --- |']
    for comparison,values in paired.items():
        for r in values:
            lines.append(f"| {comparison}/{r['seed']} | {r['oa_percent']:+.3f} | {r['aa_percent']:+.3f} | {r['kappa']:+.4f} |")
    lines+=['','Full configs, masks, class mapping, hashes, logs and checkpoints are retained in the linked NAS result directory.',
            'Target nonzero GT masks determine candidate centers; target class values never enter training or checkpoint selection.',
            'Generic published defaults are used; undocumented dataset-specific paper settings are not assumed.',
            'No version is automatically selected and no follow-up experiments are launched.']
    report='\n'.join(lines)+'\n'
    (c.OUT/'RESULTS.md').write_text(report)
    (c.ROOT/'experiments/official_bench_v1/RESULTS.md').write_text(report)


if __name__=='__main__':main()
