# Query tokenizer + A1 class-query readout (no BiDA blocks)

Same Full Joint BN; three seeds2100/2101/2102, 200 epochs. Mean ± sample SD.
Pipeline changes readout as well as depth; it is not an isolated block-removal control.

## Fixed epoch200 primary

| Arm | OA | AA | Kappa | Class6 collapse |
| --- | ---: | ---: | ---: | ---: |
| pipeline | 78.81 ± 3.72 | 62.37 ± 13.12 | 0.604 ± 0.119 | 0/3 |
| query1 | 77.58 ± 2.62 | 64.83 ± 9.24 | 0.588 ± 0.099 | 0/3 |
| query3 | 69.94 ± 7.78 | 52.19 ± 23.68 | 0.410 ± 0.305 | 1/3 |
| bida3 | 79.50 ± 1.46 | 71.90 ± 1.72 | 0.665 ± 0.024 | 0/3 |

| Seed | OA | AA | Kappa | Class6 share | Max class share | Recall coverage | Source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 80.46 | 70.68 | 0.673 | 59.87% | 59.87% | 7/7 | 100.00 |
| 2101 | 74.55 | 47.24 | 0.467 | 83.41% | 83.41% | 6/7 | 100.00 |
| 2102 | 81.42 | 69.18 | 0.671 | 66.23% | 66.23% | 7/7 | 100.00 |

## Oracle diagnostic only

| Seed | Best epoch | OA | Associated AA | Associated Kappa |
| --- | ---: | ---: | ---: | ---: |
| 2100 | 180 | 80.82 | 70.67 | 0.678 |
| 2101 | 110 | 75.87 | 51.94 | 0.513 |
| 2102 | 50 | 82.75 | 68.72 | 0.703 |

Oracle OA 79.81 ± 3.55%. Label-selected, not a formal checkpoint or guaranteed upper bound.

## Fixed per-class recall

| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 61.94 | 61.97 | 59.04 | 81.82 | 89.74 | 90.29 | 49.98 |
| 2101 | 10.13 | 62.87 | 50.13 | 54.55 | 53.92 | 99.08 | 0.00 |
| 2102 | 65.56 | 73.94 | 44.37 | 81.82 | 93.73 | 94.29 | 30.56 |

## Fixed prediction distribution

| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1370 (2.59%) | 3242 (6.13%) | 2524 (4.77%) | 187 (0.35%) | 8834 (16.70%) | 31671 (59.87%) | 5073 (9.59%) |
| 2101 | 176 (0.33%) | 3336 (6.31%) | 1894 (3.58%) | 12 (0.02%) | 3356 (6.34%) | 44127 (83.41%) | 0 (0.00%) |
| 2102 | 1218 (2.30%) | 3989 (7.54%) | 1361 (2.57%) | 20 (0.04%) | 8613 (16.28%) | 35034 (66.23%) | 2666 (5.04%) |

## Trajectory coverage

| Seed | Class6 collapse grid points | Any-class ≥95% grid points |
| --- | ---: | ---: |
| 2100 | 0/20 | 0/20 |
| 2101 | 0/20 | 0/20 |
| 2102 | 0/20 | 0/20 |

## Mean pipeline minus controls

- query1: OA +1.24 pp, AA -2.46 pp.
- query3: OA +8.87 pp, AA +10.17 pp.
- bida3: OA -0.69 pp, AA -9.53 pp.

## Fixed target angular probe (same2048 samples as prior diagnosis)

| Seed | Stage | Norm | Direction concentration | Cos(w6) |
| --- | --- | ---: | ---: | ---: |
| 2100 | spatial_tokens | 2.699 | 0.618 | 0.093 |
| 2100 | class_reader | 4.471 | 0.649 | 0.430 |
| 2100 | classifier_z | 8.122 | 0.651 | 0.434 |
| 2101 | spatial_tokens | 3.041 | 0.861 | -0.091 |
| 2101 | class_reader | 4.215 | 0.867 | 0.630 |
| 2101 | classifier_z | 8.114 | 0.867 | 0.636 |
| 2102 | spatial_tokens | 2.752 | 0.731 | 0.001 |
| 2102 | class_reader | 4.013 | 0.708 | 0.533 |
| 2102 | classifier_z | 8.098 | 0.703 | 0.534 |

Stage cosines are model-local descriptive lenses. Pre-final-normalizer vectors are not in identical coordinates to final z; no causal attribution from cosine alone.

## Interpretation

The no-BiDA-block pipeline can achieve high fixed OA in seeds2100/2102 (80.46/81.42%), but does not reproduce the balanced three-seed stability of Full Joint + BiDA tokenizer. Its mean fixed OA is 0.69 pp lower, AA is 9.53 pp lower, and both OA/AA sample SDs are larger.
None of the60 frozen grid points cross the predefined class6-collapse threshold. Nevertheless seed2101 has 83.41% class6 predictions, zero class7 recall, and only 47.24% AA. Avoiding a95% threshold is not equivalent to recovering category coverage.
On the same fixed2048 target probe, seed2101 classifier z norm is 8.114, directional concentration is 0.867, cosine to w6 is 0.636 and class6 margin is 4.757. Strong target directional/class bias persists without any BiDA refinement block.
Therefore BiDA refinement is not required for the observed strong majority-class directional bias in a trained query pipeline. Depth3 worsens the earlier Hybrid controls, but removing it and replacing the readout does not solve the whole problem.
This experiment does not establish that the spatial query tokenizer is intrinsically wrong or that the original interface is the sole cause: the new A1 class reader, retained classifier and stem also co-adapt during training. Do not promote this pipeline to a stable primary backbone; keep Full Joint + BiDA tokenizer as reference.

## Scope and checks

Verified 480 checkpoint/prediction hashes and 12 manifests; all20 predictions per new run frozen before target metrics.
Initial retained stem/tokenizer/final-normalizer/classifier tensors match the existing per-seed Hybrid initialization. New class query/reader initialization and removal of blocks alter training RNG consumption.
New pipeline has 289,719 trainable parameters. Final classifier retains BiDA initial tensors in A1-compatible LayerNorm/linear structure.
No new loss, BN rule, residual scale, learning-rate or checkpoint tuning. Three seeds on one transfer pair do not establish general stability.
Success/failure concerns the whole tokenizer-plus-class-reader pipeline, not the spatial-query tokenizer alone. Readout and parameter-count changes confound a pure depth attribution.
