# PaviaU → PaviaC normalization transfer: fixed epoch 200

BiDA-self with original BiDA tokenizer. No SceneShift or additional losses.

| Arm | OA (%) | AA (%) | Kappa | Collapse |
| --- | ---: | ---: | ---: | ---: |
| original | 68.90 ± 0.37 | 66.38 ± 1.36 | 0.622 ± 0.007 | 0/3 |
| joint | 68.87 ± 2.73 | 65.69 ± 2.97 | 0.620 ± 0.033 | 0/3 |

Mean ± sample std (ddof=1), seeds 2100/2101/2102.

## Paired Joint−Original

| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |
| --- | ---: | ---: | ---: |
| 2100 | +1.77 | +0.46 | +0.017 |
| 2101 | -3.04 | -3.20 | -0.037 |
| 2102 | +1.17 | +0.67 | +0.013 |
| Mean ± std | -0.03 ± 2.62 | -0.69 ± 2.18 | -0.002 ± 0.030 |

Predeclared positive-replication gate: **FAIL**.
The gate requires ≥2/3 positive paired seeds and ≥1 pp mean improvement in the same OA/AA metric, with no increase in collapse.
Three seeds do not establish statistical significance.

## Individual results

| Arm | Seed | OA | AA | Kappa | Class-6 share (%) | Max class share (%) | Zero-recall classes | Source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| original | 2100 | 69.29 | 67.95 | 0.630 | 16.53 | 34.37 | none | 99.90 |
| original | 2101 | 68.85 | 65.71 | 0.620 | 10.60 | 38.05 | [3] | 98.62 |
| original | 2102 | 68.56 | 65.48 | 0.617 | 10.17 | 38.34 | [3] | 100.00 |
| joint | 2100 | 71.06 | 68.40 | 0.647 | 8.85 | 40.93 | none | 100.00 |
| joint | 2101 | 65.81 | 62.51 | 0.584 | 10.34 | 39.77 | [3] | 99.69 |
| joint | 2102 | 69.73 | 66.15 | 0.630 | 7.44 | 41.54 | none | 100.00 |

## original: per-class recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 93.10 | 97.70 | 19.33 | 26.05 | 100.00 | 94.37 | 45.08 |
| 2101 | 96.62 | 97.79 | 0.00 | 1.24 | 99.69 | 87.93 | 76.72 |
| 2102 | 94.58 | 95.57 | 0.00 | 1.40 | 99.90 | 85.63 | 81.26 |

Prediction distribution (%), classes 1–7:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 18.64 | 34.37 | 2.40 | 4.83 | 9.23 | 16.53 | 14.00 |
| 2101 | 19.51 | 38.05 | 1.17 | 0.23 | 8.75 | 10.60 | 21.70 |
| 2102 | 19.22 | 38.34 | 1.33 | 0.26 | 8.92 | 10.17 | 21.76 |

## joint: per-class recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 98.28 | 96.45 | 18.29 | 13.64 | 99.86 | 79.87 | 72.43 |
| 2101 | 97.82 | 94.36 | 0.00 | 1.28 | 100.00 | 77.31 | 66.80 |
| 2102 | 97.04 | 96.40 | 1.97 | 0.55 | 100.00 | 80.13 | 86.95 |

Prediction distribution (%), classes 1–7:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 20.78 | 40.93 | 2.89 | 2.53 | 10.02 | 8.85 | 14.01 |
| 2101 | 20.48 | 39.77 | 1.69 | 0.63 | 10.75 | 10.34 | 16.34 |
| 2102 | 20.25 | 41.54 | 2.21 | 0.10 | 9.43 | 7.44 | 19.03 |

## Cost and protocol audit

| Arm | Parameters | Training seconds | Source-val OA |
| --- | ---: | ---: | ---: |
| original | 625399 | 3029.07 ± 23.29 | 99.51 ± 0.77 |
| joint | 625399 | 3365.28 ± 35.58 | 99.90 ± 0.18 |

Paired wall-time ratio Joint/Original: 1.111 ± 0.016; concurrent GPU jobs, not an isolated speed benchmark.
Dataset sample counts: `{'source_train': 37236, 'source_val': 1960, 'target_train': 39348, 'target_eval': 39348}`.
Updates per epoch: 290.
Initial tensors, all training batches and pre-forward CPU/CUDA RNG hashes match within each paired seed for all 200 epochs.
All 120 checkpoint hashes and six frozen target-prediction hashes verified.
This preserves the Houston 95% source split, normband, and target-mask sampler; it is not an official Pavia protocol reproduction.
Only the 102-band input dimension is adapted. No target-based checkpoint selection or tuning.
Full artifacts and machine-readable metrics: `results/pavia_joint_v1/summary.json`.
