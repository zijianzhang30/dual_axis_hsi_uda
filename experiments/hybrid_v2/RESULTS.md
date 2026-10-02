# Hybrid-V2: three-seed results

BiDA stem + Full Joint BN + existing A1 query tokenizer package; depth 3,
dropout 0.1, source self CE, 200 epochs. Seeds 2100/2101/2102.
Mean ± sample SD. No checkpoint selection or tuning from target labels.

## Primary: fixed epoch 200

| Model | OA | AA | Kappa | Class-6 collapse |
| --- | ---: | ---: | ---: | ---: |
| Full Joint + BiDA tokenizer | 79.50 ± 1.46 | 71.90 ± 1.72 | 0.665 ± 0.024 | 0/3 |
| Hybrid-V2 | 69.94 ± 7.78 | 52.19 ± 23.68 | 0.410 ± 0.305 | 1/3 |

Predeclared gate passes: False. Mean fixed OA change: -9.56 pp; mean AA change: -19.71 pp.

| Seed | OA | AA | Kappa | OA vs control | AA vs control | Class-6 share | Max class share | Recall coverage | Source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 78.06 | 70.17 | 0.646 | -0.50 | +0.14 | 55.81% | 55.81% | 7/7 | 100.00 |
| 2101 | 62.55 | 25.37 | 0.066 | -16.22 | -46.88 | 97.90% | 97.90% | 4/7 | 100.00 |
| 2102 | 69.21 | 61.05 | 0.517 | -11.98 | -12.37 | 52.04% | 52.04% | 7/7 | 100.00 |

## Secondary: target-oracle diagnostics only

Highest target OA among epochs 10,20,...,200; earliest epoch wins ties.
Not a target-blind estimate and not the formal selected checkpoint.

| Seed | Best epoch | OA | Corresponding AA | Corresponding Kappa | OA minus fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2100 | 120 | 80.04 | 70.93 | 0.672 | +1.98 |
| 2101 | 40 | 70.44 | 45.00 | 0.346 | +7.88 |
| 2102 | 70 | 75.04 | 65.54 | 0.580 | +5.83 |

Oracle OA: 75.17 ± 4.80%. This is a label-selected diagnostic, not a guaranteed potential ceiling.

## Fixed per-class recall (%)

| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 56.32 | 71.36 | 57.08 | 86.36 | 96.41 | 86.78 | 36.87 |
| 2101 | 0.00 | 0.00 | 2.54 | 59.09 | 16.33 | 99.61 | 0.00 |
| 2102 | 54.69 | 71.98 | 29.77 | 81.82 | 99.91 | 79.29 | 9.87 |
| Mean ± SD | 37.00 ± 32.06 | 47.78 ± 41.38 | 29.80 ± 27.27 | 75.76 ± 14.61 | 70.88 ± 47.28 | 88.56 ± 10.28 | 15.58 ± 19.09 |

## Fixed prediction distribution

| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1099 (2.08%) | 3963 (7.49%) | 2218 (4.19%) | 976 (1.84%) | 11917 (22.53%) | 29526 (55.81%) | 3202 (6.05%) |
| 2101 | 0 (0.00%) | 1 (0.00%) | 70 (0.13%) | 34 (0.06%) | 1005 (1.90%) | 51791 (97.90%) | 0 (0.00%) |
| 2102 | 1017 (1.92%) | 3820 (7.22%) | 899 (1.70%) | 18 (0.03%) | 18692 (35.33%) | 27528 (52.04%) | 927 (1.75%) |

## Frozen trajectory collapse

| Seed | Class-6 collapse grid points | Any-class ≥95% grid points |
| --- | ---: | ---: |
| 2100 | 0/20 | 0/20 |
| 2101 | 12/20 | 12/20 |
| 2102 | 0/20 | 0/20 |

## Interpretation

This experiment does not confirm complementarity: all three fixed-200 OA endpoints are below the paired Full Joint + BiDA-tokenizer controls, and seed 2101 collapses.
Seed 2101 first crosses the class-6-collapse threshold at epoch 60 and ends with 97.90% class-6 predictions and zero recall for classes 1, 2 and 7. Its oracle OA is only 70.44%, so late checkpoint selection alone does not explain the failure.
Seed 2102 remains below its paired control throughout the sampled trajectory, even at its own target-oracle endpoint. All three final source validation accuracies are 100%, again showing that source validation does not guarantee target coverage.
Full Joint stabilization is established for the tested BiDA tokenizer, not universally for every tokenizer. Retain Full Joint + BiDA tokenizer as the reference; do not promote Hybrid-V2 to the main backbone.
The specific cause of query-package instability is not identified by this combination experiment. No additional architecture or hyperparameter search was performed.

## Reproducibility and scope

Verified 120 Hybrid checkpoint/prediction hashes plus all Hybrid and paired control manifests.
All three original BiDA initial hashes and existing Hybrid transplant hashes match. Full Joint installation changes no initial tensor.
All 20 target predictions were frozen before target metrics. Full per-epoch metrics, confusion matrices and entropy are in results/hybrid_v2/summary.json.
The intervention includes query, positional encoding, cross-attention, FFN and changed parameter count. It does not isolate learnable queries alone.
Three optimization seeds on one scene pair do not establish cross-dataset stability. Zero class-6 collapse does not imply stable minority-class recall.
