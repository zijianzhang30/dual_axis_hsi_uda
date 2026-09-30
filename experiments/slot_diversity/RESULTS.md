# A1 spatial attention diversity gate

Houston13 → Houston18; fixed epoch 200, three optimization seeds.
D0 is fresh A1; D1 adds source attention diversity with fixed weight 0.1.
The target GT mask defines eligible target training centers under the Strict BiDA
protocol; target label values enter neither loss nor checkpoint selection.

| Arm | Target OA (%) | Target AA (%) | Kappa |
| --- | ---: | ---: | ---: |
| D0 | 71.08 ± 1.22 | 57.89 ± 1.93 | 0.44 ± 0.02 |
| D1 | 69.97 ± 3.19 | 56.14 ± 8.47 | 0.40 ± 0.12 |

Paired target OA difference D1 − D0: **-1.10 ± 2.01 points**.

## Per-seed outcome and slot separation

Cosines are mean off-diagonal values across the four slots.

| Arm | Seed | Source val OA | Target OA | Target AA | Source attention cosine | Target attention cosine | Source token cosine | Target token cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 2100 | 100.00 | 71.97 | 56.37 | 0.972 | 0.972 | 0.981 | 0.983 |
| D0 | 2101 | 100.00 | 71.57 | 60.07 | 0.977 | 0.973 | 0.967 | 0.970 |
| D0 | 2102 | 100.00 | 69.69 | 57.22 | 0.912 | 0.933 | 0.933 | 0.936 |
| D1 | 2100 | 100.00 | 71.71 | 55.99 | 0.048 | 0.049 | 0.312 | 0.389 |
| D1 | 2101 | 100.00 | 71.92 | 64.69 | 0.049 | 0.058 | 0.270 | 0.392 |
| D1 | 2102 | 100.00 | 66.30 | 47.74 | 0.050 | 0.056 | 0.343 | 0.487 |

## Attention location audit

The border is the outermost row or column of the 13×13 patch; it covers
48/169 = 28.4% of spatial positions. Values are fractions of target
attention mass in that border, by query slot.

| Arm | Seed | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 2100 | 0.242 | 0.233 | 0.245 | 0.227 | 0.237 |
| D0 | 2101 | 0.219 | 0.240 | 0.183 | 0.209 | 0.213 |
| D0 | 2102 | 0.220 | 0.302 | 0.295 | 0.294 | 0.278 |
| D1 | 2100 | 0.903 | 0.358 | 0.835 | 0.413 | 0.627 |
| D1 | 2101 | 0.711 | 0.826 | 0.067 | 0.825 | 0.607 |
| D1 | 2102 | 0.050 | 0.851 | 0.730 | 0.778 | 0.602 |

## Per-class target recall (%)

Mean ± sample SD across seeds; classes follow the BiDA loader order.

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | 63.09 ± 13.38 | 40.88 ± 9.73 | 53.31 ± 3.21 | 100.00 ± 0.00 | 53.50 ± 8.61 | 94.22 ± 2.04 | 0.20 ± 0.34 |
| D1 | 58.86 ± 17.53 | 38.52 ± 30.25 | 57.44 ± 1.77 | 100.00 ± 0.00 | 43.91 ± 16.31 | 94.20 ± 3.93 | 0.05 ± 0.09 |

## Training loss diagnostics

Diversity is the raw mean squared off-diagonal attention cosine.

| Arm | First diversity | Final diversity | Final source CE |
| --- | ---: | ---: | ---: |
| D0 | 0.9625 | 0.9157 | 0.0011 |
| D1 | 0.9542 | 0.0035 | 0.0014 |

Per-run summaries contain confusion matrices, per-class recall, prediction
counts, average spatial attention maps, and pairwise slot similarity matrices.
Attention-map figures for all seeds are in `results/slot_diversity/attention_maps_<seed>.png`.
The target class-example figure for seed 2100 uses GT only for post-hoc
sample selection: `results/slot_diversity/class_examples_d1_2100.png`.

## Interpretation

D1 sharply lowers attention-map and token cosine on source and target while
retaining perfect source validation OA at epoch 200. Mean target OA is 1.10
points below the fresh D0 control, with a 3.39-point drop in seed 2102.
Many D1 queries focus on fixed border rows or columns, as the maps and
border-mass audit show. This establishes spatial separation but does not
establish different semantic roles. The proposed specialization gate is
therefore only partially met; proceeding directly to same-slot cross-domain
attention would assume semantics that this experiment has not verified.
