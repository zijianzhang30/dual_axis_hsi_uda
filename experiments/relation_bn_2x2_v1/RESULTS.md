# Tokenizer × BN 2×2: Houston13 → Houston18

Fixed epoch 200 is primary. Mean ± sample std(ddof=1), seeds2100/2101/2102.
Only cell2 adds training; cells1/3/4 are audited historical reuse. No new modules or losses.

| Cell | OA (%) | AA (%) | Kappa | Class-6 collapse | Any-class collapse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1: Original BN + original tokenizer | 73.18 ± 8.21 | 54.05 ± 18.46 | 0.440 ± 0.280 | 1/3 | 1/3 |
| 2: Original BN + center-relative | 75.18 ± 10.55 | 57.61 ± 24.57 | 0.476 ± 0.339 | 1/3 | 1/3 |
| 3: Full Joint BN + original tokenizer | 79.50 ± 1.46 | 71.90 ± 1.72 | 0.665 ± 0.024 | 0/3 | 0/3 |
| 4: Full Joint BN + center-relative | 82.03 ± 0.55 | 77.01 ± 2.67 | 0.706 ± 0.010 | 0/3 | 0/3 |

## Per-seed fixed200

| Cell | Seed | OA | AA | Kappa | Class-6 share (%) | Zero-recall classes | Source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 2100 | 78.41 | 66.32 | 0.627 | 63.66 | none | 79.53 |
| 1 | 2101 | 63.72 | 32.82 | 0.119 | 95.93 | [2, 7] | 74.02 |
| 1 | 2102 | 77.40 | 63.01 | 0.575 | 72.88 | none | 67.72 |
| 2 | 2100 | 82.11 | 74.86 | 0.694 | 62.93 | none | 74.80 |
| 2 | 2101 | 63.03 | 29.48 | 0.085 | 97.26 | [7] | 70.87 |
| 2 | 2102 | 80.38 | 68.50 | 0.649 | 67.52 | none | 70.08 |
| 3 | 2100 | 78.55 | 70.03 | 0.663 | 52.02 | none | 100.00 |
| 3 | 2101 | 78.77 | 72.25 | 0.641 | 61.46 | none | 100.00 |
| 3 | 2102 | 81.19 | 73.42 | 0.690 | 58.16 | none | 100.00 |
| 4 | 2100 | 82.36 | 74.42 | 0.709 | 57.88 | none | 100.00 |
| 4 | 2101 | 81.40 | 76.85 | 0.695 | 57.72 | none | 100.00 |
| 4 | 2102 | 82.34 | 79.75 | 0.714 | 56.08 | none | 100.00 |

## Paired 2-1

| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |
| --- | ---: | ---: | ---: |
| 2100 | +3.705 | +8.531 | +0.067 |
| 2101 | -0.682 | -3.338 | -0.034 |
| 2102 | +2.975 | +5.495 | +0.074 |
| Mean ± std | 2.00 ± 2.35 | 3.56 ± 6.17 | 0.036 ± 0.060 |

## Paired 4-2

| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |
| --- | ---: | ---: | ---: |
| 2100 | +0.248 | -0.431 | +0.016 |
| 2101 | +18.366 | +47.373 | +0.609 |
| 2102 | +1.958 | +11.248 | +0.065 |
| Mean ± std | 6.86 ± 10.00 | 19.40 ± 24.92 | 0.230 ± 0.329 |

## Target-oracle diagnostic, NOT primary or target-blind selection

Max target OA among epochs10,20,...,200, earliest epoch on ties; AA/Kappa are from the same checkpoint.
Official local BiDA code additionally evaluates epoch1; that epoch is not part of this agreed grid.

| Cell | Seed | Oracle epoch | Oracle OA | Associated AA | Associated Kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2100 | 40 | 81.91 | 73.66 | 0.697 |
| 1 | 2101 | 180 | 78.36 | 65.67 | 0.614 |
| 1 | 2102 | 30 | 79.61 | 67.48 | 0.671 |
| 1 mean ± std | — | — | 79.96 ± 1.80 | 68.94 ± 4.19 | 0.660 ± 0.043 |
| 2 | 2100 | 160 | 83.02 | 74.96 | 0.711 |
| 2 | 2101 | 20 | 77.63 | 65.07 | 0.569 |
| 2 | 2102 | 130 | 82.50 | 74.66 | 0.701 |
| 2 mean ± std | — | — | 81.05 ± 2.97 | 71.56 ± 5.62 | 0.661 ± 0.079 |
| 3 | 2100 | 20 | 82.71 | 70.75 | 0.706 |
| 3 | 2101 | 60 | 80.30 | 71.96 | 0.663 |
| 3 | 2102 | 40 | 81.77 | 75.01 | 0.696 |
| 3 mean ± std | — | — | 81.59 ± 1.22 | 72.58 ± 2.19 | 0.688 ± 0.023 |
| 4 | 2100 | 120 | 83.32 | 71.47 | 0.718 |
| 4 | 2101 | 30 | 83.77 | 75.51 | 0.723 |
| 4 | 2102 | 70 | 83.62 | 77.07 | 0.729 |
| 4 mean ± std | — | — | 83.57 ± 0.23 | 74.68 ± 2.89 | 0.723 ± 0.005 |

## Cell1 fixed200 Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 40.58 | 77.24 | 57.81 | 81.82 | 93.73 | 90.26 | 22.84 |
| 2101 | 4.51 | 0.00 | 29.05 | 81.82 | 15.09 | 99.25 | 0.00 |
| 2102 | 56.61 | 74.76 | 50.27 | 81.82 | 81.59 | 95.28 | 0.73 |

## Cell1 fixed200 Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1.43 | 8.11 | 4.77 | 0.10 | 18.57 | 63.66 | 3.36 |
| 2101 | 0.21 | 0.00 | 1.73 | 0.03 | 2.09 | 95.93 | 0.00 |
| 2102 | 1.96 | 8.02 | 3.42 | 0.03 | 13.44 | 72.88 | 0.25 |

## Cell2 fixed200 Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 69.84 | 82.62 | 68.85 | 90.91 | 83.39 | 92.50 | 35.87 |
| 2101 | 0.67 | 0.59 | 3.73 | 81.82 | 19.97 | 99.57 | 0.00 |
| 2102 | 52.40 | 81.86 | 58.93 | 86.36 | 89.22 | 94.16 | 16.60 |

## Cell2 fixed200 Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 2.50 | 8.81 | 6.73 | 0.04 | 13.58 | 62.93 | 5.40 |
| 2101 | 0.02 | 0.05 | 0.20 | 0.03 | 2.43 | 97.26 | 0.00 |
| 2102 | 1.74 | 9.35 | 4.08 | 0.06 | 14.80 | 67.52 | 2.45 |

## Cell3 fixed200 Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 30.75 | 73.66 | 60.12 | 81.82 | 93.65 | 82.49 | 67.75 |
| 2101 | 74.94 | 76.06 | 63.78 | 86.36 | 92.88 | 89.29 | 22.43 |
| 2102 | 71.18 | 79.77 | 50.85 | 81.82 | 96.81 | 89.01 | 44.49 |

## Cell3 fixed200 Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 0.96 | 8.15 | 5.45 | 0.14 | 19.58 | 52.02 | 13.70 |
| 2101 | 3.26 | 7.86 | 6.24 | 0.10 | 17.47 | 61.46 | 3.62 |
| 2102 | 3.06 | 8.56 | 3.32 | 0.21 | 19.06 | 58.16 | 7.63 |

## Cell4 fixed200 Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 39.99 | 73.62 | 64.87 | 100.00 | 90.68 | 89.48 | 62.33 |
| 2101 | 61.49 | 78.80 | 66.28 | 100.00 | 92.77 | 88.06 | 50.56 |
| 2102 | 72.06 | 81.94 | 65.52 | 100.00 | 96.62 | 87.27 | 54.86 |

## Cell4 fixed200 Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1.22 | 7.67 | 5.24 | 0.12 | 15.87 | 57.88 | 11.99 |
| 2101 | 2.10 | 8.40 | 5.36 | 0.18 | 17.63 | 57.72 | 8.61 |
| 2102 | 2.52 | 8.77 | 4.72 | 0.09 | 18.97 | 56.08 | 8.84 |

## Costs and reproducibility

| Cell | Parameters | Training seconds |
| --- | ---: | ---: |
| 1 | 376567 | 203.36 ± 3.56 |
| 2 | 378011 | 158.94 ± 10.92 |
| 3 | 376567 | 193.88 ± 12.37 |
| 4 | 378011 | 197.77 ± 12.75 |

Historical/new timings use different concurrent loads, not a controlled speed comparison.
Native original BiDA source→target forward and BatchNorm counters verified; no replacement BN for cell2.
All 200 epochs match same-seed batch/RNG streams across four cells; center-MLP initialization identical in cells2/4.
All 240 grid predictions/checkpoints hashed and frozen before target scoring. Reused fixed200 metrics reproduce exactly.
No target-based tuning, early stopping, formal checkpoint selection or automatic version promotion.
Pavia and other historical artifacts were not modified. No additional training automatically appended.
Inference encountered root-disk exhaustion near completion. Only this new result directory was moved to NAS, with a project symlink retained. Completed predictions were preserved; invalid partial writes were quarantined and missing predictions resumed without retraining.
Physical result directory: `/nas1/zhangzj26/dual_axis_hsi_uda/results/relation_bn_2x2_v1`.
Configs/logs/code snapshots/reuse audit/full trajectories/paired deltas: `results/relation_bn_2x2_v1/summary.json`.
