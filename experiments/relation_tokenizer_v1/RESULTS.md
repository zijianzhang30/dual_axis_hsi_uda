# Relation tokenizer V1: Houston13 → Houston18

Fixed epoch 200, seeds 2100/2101/2102. Mean ± sample std (ddof=1).
A: original weights; B: local-standardized relation weights; C: center-relative relation weights.
All use unchanged CNN values, Full Joint BN, depth3 self Transformer and source CE only.

| Arm | OA (%) | AA (%) | Kappa | Class-6 collapse | Any-class collapse |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 79.50 ± 1.46 | 71.90 ± 1.72 | 0.665 ± 0.024 | 0/3 | 0/3 |
| B | 80.82 ± 0.52 | 71.16 ± 0.66 | 0.672 ± 0.012 | 0/3 | 0/3 |
| C | 82.03 ± 0.55 | 77.01 ± 2.67 | 0.706 ± 0.010 | 0/3 | 0/3 |

## Individual endpoints

| Arm | Seed | OA | AA | Kappa | Class-6 share (%) | Zero-recall classes | Source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A | 2100 | 78.55 | 70.03 | 0.663 | 52.02 | none | 100.00 |
| A | 2101 | 78.77 | 72.25 | 0.641 | 61.46 | none | 100.00 |
| A | 2102 | 81.19 | 73.42 | 0.690 | 58.16 | none | 100.00 |
| B | 2100 | 80.54 | 70.47 | 0.659 | 65.03 | none | 100.00 |
| B | 2101 | 81.42 | 71.19 | 0.675 | 65.12 | none | 100.00 |
| B | 2102 | 80.50 | 71.80 | 0.682 | 56.72 | none | 100.00 |
| C | 2100 | 82.36 | 74.42 | 0.709 | 57.88 | none | 100.00 |
| C | 2101 | 81.40 | 76.85 | 0.695 | 57.72 | none | 100.00 |
| C | 2102 | 82.34 | 79.75 | 0.714 | 56.08 | none | 100.00 |

## Paired C-A

| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |
| --- | ---: | ---: | ---: |
| 2100 | +3.809 | +4.391 | +0.046 |
| 2101 | +2.631 | +4.602 | +0.053 |
| 2102 | +1.149 | +6.335 | +0.024 |
| Mean ± std | 2.53 ± 1.33 | 5.11 ± 1.07 | 0.041 ± 0.015 |

## Paired C-B

| Seed | ΔOA (pp) | ΔAA (pp) | ΔKappa |
| --- | ---: | ---: | ---: |
| 2100 | +1.822 | +3.951 | +0.050 |
| 2101 | -0.015 | +5.657 | +0.019 |
| 2102 | +1.843 | +7.952 | +0.032 |
| Mean ± std | 1.22 ± 1.07 | 5.85 ± 2.01 | 0.034 ± 0.016 |

## A: Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 30.75 | 73.66 | 60.12 | 81.82 | 93.65 | 82.49 | 67.75 |
| 2101 | 74.94 | 76.06 | 63.78 | 86.36 | 92.88 | 89.29 | 22.43 |
| 2102 | 71.18 | 79.77 | 50.85 | 81.82 | 96.81 | 89.01 | 44.49 |

## A: Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 0.96 | 8.15 | 5.45 | 0.14 | 19.58 | 52.02 | 13.70 |
| 2101 | 3.26 | 7.86 | 6.24 | 0.10 | 17.47 | 61.46 | 3.62 |
| 2102 | 3.06 | 8.56 | 3.32 | 0.21 | 19.06 | 58.16 | 7.63 |

## B: Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 62.90 | 61.71 | 52.44 | 100.00 | 65.08 | 93.54 | 57.65 |
| 2101 | 67.92 | 71.03 | 65.41 | 81.82 | 83.45 | 93.63 | 35.10 |
| 2102 | 51.44 | 75.13 | 59.76 | 81.82 | 95.42 | 87.49 | 51.54 |

## B: Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 2.59 | 6.81 | 3.76 | 0.09 | 10.27 | 65.03 | 11.45 |
| 2101 | 2.88 | 7.36 | 6.00 | 0.11 | 12.42 | 65.12 | 6.12 |
| 2102 | 1.80 | 8.25 | 5.08 | 0.06 | 18.97 | 56.72 | 9.13 |

## C: Recall (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 39.99 | 73.62 | 64.87 | 100.00 | 90.68 | 89.48 | 62.33 |
| 2101 | 61.49 | 78.80 | 66.28 | 100.00 | 92.77 | 88.06 | 50.56 |
| 2102 | 72.06 | 81.94 | 65.52 | 100.00 | 96.62 | 87.27 | 54.86 |

## C: Prediction share (%)

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1.22 | 7.67 | 5.24 | 0.12 | 15.87 | 57.88 | 11.99 |
| 2101 | 2.10 | 8.40 | 5.36 | 0.18 | 17.63 | 57.72 | 8.61 |
| 2102 | 2.52 | 8.77 | 4.72 | 0.09 | 18.97 | 56.08 | 8.84 |

## Parameters, cost and audit

| Arm | Parameters | Training seconds | Source-val OA |
| --- | ---: | ---: | ---: |
| A | 376567 | 193.88 ± 12.37 | 100.00 ± 0.00 |
| B | 378011 | 198.61 ± 12.64 | 100.00 ± 0.00 |
| C | 378011 | 197.77 ± 12.75 | 100.00 ± 0.00 |

Concurrent GPU wall times are observational, not an isolated speed benchmark.
B/C replace the 256-parameter conv_a with a 1700-parameter pointwise MLP; +1444 parameters.
Implementation numerical, spatial alignment, finite-gradient and target-joint-statistics gradient checks passed.
A freshly replayed all three historical final models exactly, including BN buffers.
All three arms match same-seed batch and pre-forward RNG hashes for all 200 epochs.
All 180 checkpoint and nine prediction hashes verified before target scoring.
No Pavia files or processes changed; no target-based tuning, early stopping or version selection.
No new experiments are automatically appended from these results.
Full configs/logs/manifests/code snapshots/diff and summary: `results/relation_tokenizer_v1/`.
