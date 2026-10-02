# Key-only spatial PE: three-seed fixed-endpoint comparison

Completed seeds 2100/2101/2102, 200 epochs. Seed 2101 ran on GPU 5;
2100 and 2102 ran concurrently on GPU 4. No other variant was trained.
Depth 0 + Class Query + Full Joint BN retained. The spatial reader alone uses
Q=q_norm(query), K=shared_kv_norm(X+PE), V=shared_kv_norm(X); attention
projections, residual, FFN, class reader and all trainable tensors are unchanged
at initialization (289,719 parameters). Native attention dropout remains zero,
token dropout 0.1, source CE only and SGD 0.01.

Numerical tests passed: PE=0 exact original-forward equivalence, unchanged
parameter state and CPU RNG, unchanged class-reader forward, hooked K/V inputs
matching the intended expressions, and finite nonzero input/query gradients.
All original/transplant/final-pipeline initialization hashes and shared code
hashes matched each paired control. Historical batch hashes do not exist;
same loader/RNG path is maintained by construction, not direct all-batch audit.

## Primary: fixed epoch 200

| Seed | OA | AA | Kappa | Class-6 prediction share | Zero-recall classes |
| --- | ---: | ---: | ---: | ---: | --- |
| 2100 | 80.55% | 64.75% | 0.659 | 64.95% | None |
| 2101 | 79.27% | 63.97% | 0.657 | 59.12% | None |
| 2102 | 79.83% | 67.26% | 0.634 | 69.10% | None |

Sample SD uses ddof=1 across the three seeds.

| Model | OA mean +/- SD | AA mean +/- SD | Kappa mean +/- SD | Class-6 collapse |
| --- | ---: | ---: | ---: | ---: |
| Query + Class Query PE-on, depth 0 | 78.81 +/- 3.72% | 62.37 +/- 13.12% | 0.604 +/- 0.119 | 0/3 |
| Query + Class Query PE-off, depth 0 | 79.54 +/- 2.02% | 67.25 +/- 3.06% | 0.652 +/- 0.028 | 0/3 |
| Query + Class Query Key-only PE, depth 0 | 79.89 +/- 0.64% | 65.33 +/- 1.72% | 0.650 +/- 0.014 | 0/3 |
| Full Joint + BiDA tokenizer, depth 3 reference | 79.50 +/- 1.46% | 71.90 +/- 1.72% | 0.665 +/- 0.024 | 0/3 |

Collapse uses the previously fixed >=95% class-6 prediction share threshold.
All three Query depth-0 variants already have 0/3 under this threshold; do not
claim Key-only PE reduces thresholded collapse count versus PE-on. It retains
nonzero fixed-endpoint recall for all classes. Small OA SD does not mean each
minority-class recall is stable, and three seeds are limited evidence.

## Paired effects

| Seed | OA vs PE-on | AA vs PE-on | OA vs PE-off | AA vs PE-off |
| --- | ---: | ---: | ---: | ---: |
| 2100 | +0.09 pp | -5.94 pp | +2.71 pp | -0.76 pp |
| 2101 | +4.72 pp | +16.74 pp | +0.25 pp | -1.49 pp |
| 2102 | -1.59 pp | -1.92 pp | -1.94 pp | -3.53 pp |

Key-only minus PE-off mean OA is +0.34 pp and AA is -1.92 pp; AA is lower in
every paired seed. Key-only minus BiDA reference mean OA is +0.38 pp and AA is
-6.57 pp. Reference depth/readout differ, so its comparison is a practical
backbone benchmark, not a strict single-component tokenizer causal contrast.

## Fixed-endpoint per-class recall

| Class | 2100 | 2101 | 2102 |
| --- | ---: | ---: | ---: |
| 1 | 22.91% | 90.98% | 61.12% |
| 2 | 70.85% | 65.99% | 81.84% |
| 3 | 64.40% | 58.24% | 56.39% |
| 4 | 81.82% | 22.73% | 81.82% |
| 5 | 65.34% | 74.96% | 73.94% |
| 6 | 93.41% | 90.06% | 94.79% |
| 7 | 54.49% | 44.86% | 20.94% |

The reference's mean class-5 recall is 94.45%, versus 71.41% for Key-only;
class-4 mean is 83.33% versus 62.12%. OA alone hides these class-balanced costs.

## Fixed-endpoint prediction counts

Each seed predicts 52,901 samples; exact shares and confusion matrices are
preserved in results JSONs and the aggregate summary.

| Predicted class | 2100 | 2101 | 2102 |
| --- | ---: | ---: | ---: |
| 1 | 411 | 3328 | 1159 |
| 2 | 4088 | 5427 | 4749 |
| 3 | 3317 | 2173 | 2299 |
| 4 | 32 | 5 | 27 |
| 5 | 5480 | 6791 | 6385 |
| 6 | 34360 | 31277 | 36556 |
| 7 | 5213 | 3900 | 1726 |

## Oracle diagnostic, not checkpoint selection

| Seed | Best grid OA | Epoch | Minimum grid OA |
| --- | ---: | ---: | ---: |
| 2100 | 81.06% | 130 | 76.12% |
| 2101 | 79.72% | 170 | 76.03% |
| 2102 | 80.70% | 30 | 79.01% |

Oracle OA is 80.49 +/- 0.69%, diagnostic only. None of the 60 saved Key-only
checkpoints reaches class-6-collapse threshold. Seed 2101 has zero-recall
classes at epochs 10,20,30,40,50,70,120, so full coverage is not guaranteed
throughout training. All metrics were collected after freezing/hash recording
of all predictions for each run; fixed epoch 200 remains the formal endpoint.

## Decision

The user's practical joint gate, evaluated against unrounded reference means:

- Mean OA >= reference: passes (79.8857 vs 79.5039).
- Mean AA >= reference: fails (65.3276 vs 71.8998).
- 0/3 thresholded class-6 collapse: passes.

Overall gate fails. Key-only PE is not promoted to the backbone or a confirmed
second innovation. In these seeds it has lower OA dispersion, but not superior
class-balanced performance. This intervention does not establish that removing
PE from Value recovers the tokenizer's missing semantic coverage; it also does
not refute every PE/attention mechanism. No new direction probe was run, so do
not claim measured improvement in concentration for Key-only based only on OA.

Per the user's stopping rule, stop expanding the tokenizer branch and retain
Full Joint + BiDA tokenizer as main reference. No LN/attention rescue sweep,
new module, or UDA loss was started.

## Does the current method beat the original base?

Yes, if base means original Mixed-BN BiDA-self under the same three-seed,
fixed-200 reporting: OA 73.18 +/- 8.21 -> 79.50 +/- 1.46% (+6.33 pp), AA
54.05 +/- 18.46 -> 71.90 +/- 1.72% (+17.85 pp). The defensible improvement
is Full Joint normalization, not a demonstrated extra tokenizer advantage.
These are not claims of superiority to official oracle results or other
methods evaluated with a different protocol.

## Artifacts

`results/query_key_only_v1/formal_{2100,2101,2102}/` holds the new frozen runs.
`results/query_key_only_v1/summary.json` holds exact paired metrics, full
recalls/distributions, anomalies and gate results. All 360 checkpoint/prediction
file hashes across the nine on/off/key-only runs, all nine manifest hashes,
and paired initial/config equality checks passed. No previous artifact changed.
