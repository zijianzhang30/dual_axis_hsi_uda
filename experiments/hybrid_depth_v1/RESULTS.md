# Hybrid depth V1: fixed endpoints

Full Joint BN unchanged. Query package depth1/3 and BiDA-tokenizer depth3.
Seeds 2100/2101/2102; epoch200 primary. Mean ± sample SD.

| Arm | OA | AA | Kappa | Class6 collapse |
| --- | ---: | ---: | ---: | ---: |
| query1 | 77.58 ± 2.62 | 64.83 ± 9.24 | 0.588 ± 0.099 | 0/3 |
| query3 | 69.94 ± 7.78 | 52.19 ± 23.68 | 0.410 ± 0.305 | 1/3 |
| bida3 | 79.50 ± 1.46 | 71.90 ± 1.72 | 0.665 ± 0.024 | 0/3 |

| Arm | Seed | OA | AA | Kappa | Class6 share | Recall coverage | Oracle OA | Oracle epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query1 | 2100 | 79.96 | 69.49 | 0.654 | 64.04% | 7/7 | 81.17 | 30 |
| query1 | 2101 | 74.77 | 54.19 | 0.474 | 83.03% | 6/7 | 78.68 | 20 |
| query1 | 2102 | 77.99 | 70.81 | 0.636 | 58.68% | 7/7 | 78.00 | 170 |
| query3 | 2100 | 78.06 | 70.17 | 0.646 | 55.81% | 7/7 | 80.04 | 120 |
| query3 | 2101 | 62.55 | 25.37 | 0.066 | 97.90% | 4/7 | 70.44 | 40 |
| query3 | 2102 | 69.21 | 61.05 | 0.517 | 52.04% | 7/7 | 75.04 | 70 |

## Depth1 fixed per-class recall

| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 69.99 | 53.12 | 71.71 | 81.82 | 66.52 | 93.20 | 50.06 |
| 2101 | 36.59 | 73.35 | 50.67 | 81.82 | 37.52 | 99.39 | 0.00 |
| 2102 | 70.73 | 75.63 | 54.55 | 81.82 | 96.98 | 87.12 | 28.87 |

## Depth1 prediction distribution

| Seed | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1715 (3.24%) | 2738 (5.18%) | 4439 (8.39%) | 20 (0.04%) | 5185 (9.80%) | 33878 (64.04%) | 4926 (9.31%) |
| 2101 | 603 (1.14%) | 4198 (7.94%) | 1754 (3.32%) | 25 (0.05%) | 2398 (4.53%) | 43923 (83.03%) | 0 (0.00%) |
| 2102 | 1376 (2.60%) | 4373 (8.27%) | 1881 (3.56%) | 80 (0.15%) | 11422 (21.59%) | 31043 (58.68%) | 2726 (5.15%) |

## Paired depth1 minus depth3 changes

- 2100: OA +1.91 pp, AA -0.68 pp.
- 2101: OA +12.22 pp, AA +28.82 pp.
- 2102: OA +8.78 pp, AA +9.77 pp.

## Interpretation

Depth1 improves mean fixed OA by 7.64 pp and AA by 12.64 pp versus Query depth3; OA sample SD falls from 7.78 to 2.62. None of its 60 frozen grid checkpoints cross the class6-collapse threshold.
This is a meaningful depth-related rescue, not full stability: seed2101 still assigns 83.03% of predictions to class6 and has class7 recall zero. Mean OA/AA remain 1.93/7.07 pp below Full Joint + BiDA-tokenizer depth3, with larger across-seed AA variation.
Layer probes show elevated target CLS concentration in block1 for both trained depths, and further class6 alignment in block2 of the depth3 model. The results support depth-dependent worsening of a target representational bias; they do not establish that block1 alone is a sufficient cause or that an interface normalization patch will fix it.
Keep Full Joint + BiDA tokenizer as the main reference. Query depth1 remains an exploratory control, not a promoted backbone.

## Scope and reproducibility

Verified 360 checkpoint/prediction SHA256 entries and nine manifests.
All initial retained tensors match the full initialized Hybrid. Only blocks2/3 are removed; Full Joint BN and classifier/readout are unchanged.
Fixed200 results remain formal; oracle OA is target-selected and diagnostic only. All20 predictions were frozen before post-hoc target metrics.
Depth0 is deferred with user agreement. Three seeds/one transfer pair and removal of blocks do not uniquely isolate an interface defect. Dropout RNG consumption also changes with depth.
Stage probes are in LAYERS.md. No decorrelation loss, cosine classifier, BN change or rescue tuning was introduced.
