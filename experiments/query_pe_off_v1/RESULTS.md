# PE-off paired causal contrast: seed 2101

Only spatial sinusoidal PE addition is removed. Query tokenizer, depth 0,
Class Query, Full Joint differentiable BN, token dropout 0.1, source CE,
SGD 0.01 and 200 epochs remain fixed. All initial tensors match the existing
PE-on pipeline (289,719 trainable parameters); shared component script hashes
match. PE-on first-epoch replay source CE exactly matches the historical value
1.1029226515028212. Historical batches were not hashed: same batch path is
supported by identical seeded loaders, shapes and RNG consumption, not a full
historical batch-hash audit. No other seed or component was trained.

## Fixed epoch 200: primary result

| Metric | PE-on | PE-off | Off minus on |
| --- | ---: | ---: | ---: |
| Target OA | 74.55% | 79.02% | +4.47 pp |
| Target AA | 47.24% | 65.46% | +18.22 pp |
| Kappa | 0.467 | 0.635 | +0.169 |
| Class-6 prediction share | 83.41% | 64.83% | -18.59 pp |
| Class-7 recall | 0.00% | 22.51% | +22.51 pp |
| Source validation OA | see historical trajectory | 100.00% | not a target-selection criterion |

Both endpoints are below the pre-existing >=95% class-6-collapse threshold;
PE-on nevertheless loses class 7. PE-off has nonzero recall for all seven
classes at epoch 200. This is improved coverage, not proof of full stability.

| Class | PE-on recall | PE-off recall | PE-on predicted count | PE-off predicted count |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10.13% | 81.15% | 176 | 2294 |
| 2 | 62.87% | 78.58% | 3336 | 6604 |
| 3 | 50.13% | 62.04% | 1894 | 2501 |
| 4 | 54.55% | 54.55% | 12 | 12 |
| 5 | 53.92% | 65.70% | 3356 | 5354 |
| 6 | 99.08% | 93.71% | 44127 | 34295 |
| 7 | 0.00% | 22.51% | 0 | 1841 |

## Geometry: identical fixed unlabeled probe samples

Each model is evaluated on the same 2048 source and 2048 target patches at
all 20 saved epochs, with exact hook-on/off logits and unchanged model-state
hashes. Concentration uses each sample's spatial/query-token mean unit vector.

| Stage concentration | On: source | On: target | Off: source | Off: target |
| --- | ---: | ---: | ---: | ---: |
| Stem ReLU | 0.797 | 0.858 | 0.800 | 0.856 |
| Reader context (PE added only in on) | 0.932 | 0.969 | 0.800 | 0.856 |
| KV LayerNorm | 0.714 | 0.915 | 0.308 | 0.528 |
| Attention residual | 0.427 | 0.870 | 0.281 | 0.615 |
| FFN residual / four tokens | 0.371 | 0.861 | 0.347 | 0.663 |
| Classifier z | 0.130 | 0.867 | 0.141 | 0.708 |

Target cos(z,w6) falls 0.636 -> 0.477. Final-z cross-sample centered RMS rises
4.037 -> 5.730, and pooled covariance participation rank rises 2.932 -> 3.687.
Random target-target z cosine mean falls 0.750 -> 0.497. These changes accompany
AA/coverage improvement, rather than only removing a constant offset from the
reported raw cosine.

Attention becomes less concentrated on target, but does NOT behave exactly
like source: off KV LN -> attention is 0.528 -> 0.615 on target versus
0.308 -> 0.281 on source. Off FFN increases target concentration to 0.663.
Therefore do not say attention itself has been fixed or that all directional
bias is gone. The reader's input/LN interaction and subsequent learned weights
co-adapt during the intervention; this is not an inference-only PE toggle.

The attention source-target concentration gap across saved epochs is
0.373--0.444 on versus 0.237--0.350 off. This supports a persistent improvement
within this run, not only a fortunate epoch-200 checkpoint.

## Oracle diagnostic and trajectory

PE-on target-oracle OA is 75.87% at epoch 110. PE-off oracle OA is 80.20%
at epoch 60 (AA 62.32%), diagnostic only; fixed epoch 200 remains primary.
PE-off OA across all 20 checkpoints is 78.00--80.20%, and class-6 prediction
share is 58.63--68.12%; no checkpoint crosses the >=95% threshold. This is
within-run checkpoint behavior, not evidence of reduced cross-seed variance.

## Interpretation

In this paired stress-seed configuration, retaining spatial PE contributes to
poorer target coverage and greater directional concentration. Removing it has
a substantial positive causal intervention effect on OA/AA. Together with the
probe, this supports investigating PE x LN/query-attention interaction rather
than treating learned queries as intrinsically unsuitable.

It does not isolate a unique mathematical mechanism, establish that PE is the
only cause, or demonstrate three-seed robustness. Do not upgrade the backbone
or compare this single seed as a mean against the Full Joint + BiDA tokenizer
three-seed reference 79.50 +/- 1.46%. No follow-up ablation was started.

## Artifacts and verification

Training took 165.45 seconds on GPU 6. All 20 checkpoints and 20 prediction
artifacts were frozen before post-hoc target metrics, and all 40 file hashes
plus the manifest hash were verified. Target geometry probe ran on GPU 7.

- Results: `results/query_pe_off_v1/formal_2101/results.json`
- Internal probe: `results/query_pe_off_v1/internal_probe/query_pe_off_2101.json`
- Frozen manifest SHA256:
  `c5acd0102502f859424d169ead0d92e91a8e6813b0046b5d15971b815190dddc`
- Executed training script SHA256 (recorded in the frozen config):
  `06bfe49cac3ce34c68f9a8131ef5b51fbfef58faa25b624484d0f609ebfcd1ce`

After all artifacts and results had been saved, the terminal-only final print
raised KeyError due to a stale comparison-field name. That print was corrected
in the current script; frozen config/checkpoints/results were not rewritten.
Thus the current script hash intentionally differs from the executed hash,
with no change to training, inference, metric computation or saved outcomes.
