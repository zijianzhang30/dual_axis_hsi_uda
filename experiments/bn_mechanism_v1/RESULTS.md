# BN mechanism V1: fixed epoch-200 results

Three optimization seeds (2100/2101/2102), one Houston transfer pair. Mean ± sample SD.

| Arm | OA | AA | Class-6 collapse |
| --- | ---: | ---: | ---: |
| original_mixed | 73.18 ± 8.21 | 54.05 ± 18.46 | 1/3 |
| buffer_only | 75.14 ± 4.01 | 61.16 ± 9.99 | 0/3 |
| detach_target | 72.73 ± 4.06 | 51.71 ± 6.93 | 0/3 |
| fixed_mixture | 79.50 ± 1.46 | 71.90 ± 1.72 | 0/3 |

Balanced inference buffers alone improve mean OA by 1.96 points and remove endpoint class-6 collapse, but remain 4.37 OA and 10.74 AA points below Full Joint.
Stopping target moment gradients lowers OA relative to Full Joint in all three seeds (8.67, 7.83, 3.81 points); mean OA/AA fall by 6.77/20.19 points.
These controlled endpoints support a contribution from target-side moment gradients during source-discriminative training, beyond buffer calibration. They do not prove that shared training normalization without those gradients is beneficial: Detach underperforms Buffer-only on average.
The differences are not an additive decomposition: trained parameters and subsequent statistics co-evolve differently in each arm.

## Per-seed endpoints

| Arm | Seed | OA | AA | Class-6 share |
| --- | ---: | ---: | ---: | ---: |
| original_mixed | 2100 | 78.41 | 66.32 | 63.66% |
| original_mixed | 2101 | 63.72 | 32.82 | 95.93% |
| original_mixed | 2102 | 77.40 | 63.01 | 72.88% |
| buffer_only | 2100 | 77.51 | 68.71 | 58.21% |
| buffer_only | 2101 | 70.51 | 49.83 | 83.09% |
| buffer_only | 2102 | 77.39 | 64.94 | 67.18% |
| detach_target | 2100 | 69.88 | 48.12 | 65.81% |
| detach_target | 2101 | 70.94 | 47.31 | 80.06% |
| detach_target | 2102 | 77.37 | 59.70 | 64.26% |
| fixed_mixture | 2100 | 78.55 | 70.03 | 52.02% |
| fixed_mixture | 2101 | 78.77 | 72.25 | 61.46% |
| fixed_mixture | 2102 | 81.19 | 73.42 | 58.16% |

## Predeclared descriptive proximity to Full Joint

Not a statistical equivalence test. All three paired OA gaps and mean OA/AA gaps must be within two points, with no endpoint class-6 collapse.

- buffer_only: close = False; paired OA differences = [-1.0396778888868, -8.26071340806412, -3.7976597795882867]; mean AA difference = -10.74.
- detach_target: close = False; paired OA differences = [-8.674694240184508, -7.825939018166011, -3.8146726904973463]; mean AA difference = -20.19.

## Mean per-class recall

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_mixed | 33.90 | 50.67 | 45.71 | 81.82 | 63.47 | 94.93 | 7.86 |
| buffer_only | 49.45 | 61.35 | 45.79 | 81.82 | 88.03 | 91.32 | 10.38 |
| detach_target | 5.67 | 90.54 | 44.61 | 81.82 | 37.67 | 93.68 | 7.99 |
| fixed_mixture | 58.96 | 76.49 | 58.25 | 83.33 | 94.45 | 86.93 | 44.89 |

## Frozen trajectory collapse

| Arm | Seed | Class-6 collapse grid points |
| --- | ---: | ---: |
| buffer_only | 2100 | 0/20 |
| buffer_only | 2101 | 0/20 |
| buffer_only | 2102 | 0/20 |
| detach_target | 2100 | 0/20 |
| detach_target | 2101 | 2/20 |
| detach_target | 2102 | 0/20 |
| fixed_mixture | 2100 | 0/20 |
| fixed_mixture | 2101 | 0/20 |
| fixed_mixture | 2102 | 0/20 |

## Controls and limitations

Buffer-only source training losses and final learned tensors reproduce Original Mixed BN exactly for all three seeds. Only running buffers differ.
Detach-target has exactly Full Joint forward arithmetic in the unit fixture, but no target input gradient through the moments under source-only CE.
Both new arms use 0.19 EMA per paired step, matched to the original two 0.1 updates. Intermediate buffer-only layer moments differ from Full Joint because training normalization differs.
Verified 360 checkpoint/prediction hashes and all 12 manifests. All 20 predictions per new run were frozen before post-hoc target metrics.
Three seeds in one transfer pair do not establish a general mechanism across datasets. No oracle checkpoint or mixture-rate tuning is used.
