# Pavia epoch-100 diagnostic

User-requested mid-training diagnostic, not the primary result. Fixed epoch 200 remains primary; no target-based selection or tuning.

| Arm | OA (%) | AA (%) | Kappa | Collapse |
| --- | ---: | ---: | ---: | ---: |
| original | 68.54 ± 3.97 | 66.94 ± 2.42 | 0.620 ± 0.048 | 0/3 |
| joint | 69.94 ± 0.62 | 66.63 ± 1.33 | 0.634 ± 0.009 | 0/3 |

Mean ± sample std, ddof=1.

| Arm | Seed | OA | AA | Kappa | Class-6 share (%) | Max class share (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 2100 | 72.42 | 69.73 | 0.668 | 15.85 | 30.43 |
| original | 2101 | 64.48 | 65.35 | 0.571 | 18.65 | 40.15 |
| original | 2102 | 68.73 | 65.76 | 0.619 | 10.07 | 38.38 |
| joint | 2100 | 70.48 | 67.73 | 0.642 | 10.79 | 37.24 |
| joint | 2101 | 70.07 | 67.00 | 0.635 | 8.30 | 38.83 |
| joint | 2102 | 69.26 | 65.15 | 0.625 | 6.27 | 39.47 |

## Paired Joint−Original

| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |
| --- | ---: | ---: | ---: |
| 2100 | -1.942 | -1.994 | -0.026 |
| 2101 | +5.594 | +1.647 | +0.064 |
| 2102 | +0.529 | -0.602 | +0.006 |
| Mean ± std | 1.39 ± 3.84 | -0.32 ± 1.84 | 0.014 ± 0.046 |

## original_2100: classes 1–7

| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 95.51 | 95.48 | 13.71 | 45.11 | 100.00 | 91.81 | 46.48 |
| Prediction share | 19.20 | 30.43 | 1.85 | 8.35 | 9.99 | 15.85 | 14.33 |

## original_2101: classes 1–7

| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 89.02 | 98.60 | 29.91 | 1.47 | 99.58 | 97.57 | 41.30 |
| Prediction share | 17.34 | 40.15 | 3.08 | 0.27 | 8.29 | 18.65 | 12.22 |

## original_2102: classes 1–7

| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 94.41 | 95.25 | 0.04 | 1.37 | 99.93 | 86.96 | 82.34 |
| Prediction share | 19.15 | 38.38 | 1.95 | 0.25 | 8.86 | 10.07 | 21.33 |

## joint_2100: classes 1–7

| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 97.63 | 94.12 | 16.01 | 23.37 | 99.86 | 80.16 | 62.97 |
| Prediction share | 20.74 | 37.24 | 2.98 | 4.40 | 11.08 | 10.79 | 12.77 |

## joint_2101: classes 1–7

| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 96.09 | 96.06 | 1.04 | 0.25 | 99.83 | 88.12 | 87.59 |
| Prediction share | 19.44 | 38.83 | 2.18 | 0.05 | 9.33 | 8.30 | 21.87 |

## joint_2102: classes 1–7

| Metric (%) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 98.14 | 93.34 | 0.67 | 3.90 | 100.00 | 72.43 | 87.59 |
| Prediction share | 20.93 | 39.47 | 4.02 | 1.39 | 9.77 | 6.27 | 18.15 |

Frozen manifest SHA256: `f0dad3bf351d4f06a87ef9e78dba8f0038e0ff99830d9daaa35b8ce2134fd3e2`.
Training processes, checkpoints and the original frozen protocol were not modified.
