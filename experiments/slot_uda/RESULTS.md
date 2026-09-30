# B0–B4 Houston13 → Houston18

Primary endpoint: target self OA at fixed epoch 200; same three optimization seeds.
Target GT mask supplies training candidate centers, but target labels enter no loss or checkpoint choice.

| Arm | OA (%) | AA (%) | Kappa |
| --- | ---: | ---: | ---: |
| B0 | 70.38 ± 1.82 | 55.56 ± 4.72 | 0.40 ± 0.06 |
| B1 | 38.21 ± 25.12 | 51.11 ± 31.90 | 0.26 ± 0.22 |
| B2 | 36.69 ± 23.99 | 50.20 ± 31.11 | 0.24 ± 0.21 |
| B3 | 53.19 ± 14.07 | 54.12 ± 12.76 | 0.36 ± 0.12 |
| B4 | 38.07 ± 25.22 | 50.44 ± 31.32 | 0.26 ± 0.22 |

| Paired difference | Target OA points, mean ± SD |
| --- | ---: |
| B2 − B1 | -1.52 ± 2.63 |
| B4 − B2 | 1.38 ± 1.23 |
| B3 − B0 | -17.20 ± 12.81 |
| B4 − B3 | -15.12 ± 38.28 |

## Per-seed OA (%)

| Seed | B0 | B1 | B2 | B3 | B4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2100 | 71.52 | 53.56 | 53.56 | 50.19 | 55.92 |
| 2101 | 71.35 | 9.22 | 9.22 | 68.51 | 9.22 |
| 2102 | 68.28 | 51.84 | 47.28 | 40.85 | 49.06 |

## Per-class recall (%)

Mean ± sample SD across seeds; classes follow the BiDA loader order.

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 61.59 ± 16.15 | 30.11 ± 10.70 | 50.37 ± 4.38 | 100.00 ± 0.00 | 51.41 ± 7.85 | 95.40 ± 1.26 | 0.02 ± 0.04 |
| B1 | 52.35 ± 45.34 | 88.49 ± 10.04 | 48.92 ± 42.38 | 66.67 ± 57.74 | 37.59 ± 32.72 | 30.06 ± 26.15 | 33.70 ± 29.37 |
| B2 | 45.85 ± 40.49 | 84.17 ± 15.87 | 49.00 ± 42.44 | 66.67 ± 57.74 | 43.09 ± 38.23 | 27.25 ± 24.67 | 35.35 ± 32.69 |
| B3 | 70.76 ± 28.70 | 57.79 ± 26.51 | 53.74 ± 6.68 | 74.24 ± 44.61 | 61.63 ± 33.11 | 60.68 ± 17.11 | 0.00 ± 0.00 |
| B4 | 44.86 ± 39.32 | 88.66 ± 10.16 | 49.27 ± 42.73 | 66.67 ± 57.74 | 38.19 ± 33.35 | 29.56 ± 26.44 | 35.87 ± 31.28 |

## Representation and collapse audit

Cosines are mean off-diagonal values over the four slots, on all target patches.

| Arm | Seed | Source val OA (%) | Target slot cosine | Target attention cosine | Predicted classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 2100 | 100.00 | 0.980 | 0.969 | 6 / 7 |
| B0 | 2101 | 100.00 | 0.971 | 0.976 | 7 / 7 |
| B0 | 2102 | 100.00 | 0.940 | 0.934 | 7 / 7 |
| B1 | 2100 | 100.00 | 0.853 | 0.922 | 7 / 7 |
| B1 | 2101 | 14.17 | 0.996 | 0.998 | 1 / 7 |
| B1 | 2102 | 100.00 | 0.792 | 0.946 | 7 / 7 |
| B2 | 2100 | 100.00 | 0.978 | 0.992 | 7 / 7 |
| B2 | 2101 | 14.17 | 0.996 | 0.999 | 1 / 7 |
| B2 | 2102 | 96.06 | 0.994 | 0.995 | 7 / 7 |
| B3 | 2100 | 100.00 | 0.977 | 0.963 | 6 / 7 |
| B3 | 2101 | 100.00 | 0.975 | 0.976 | 6 / 7 |
| B3 | 2102 | 100.00 | 0.957 | 0.951 | 6 / 7 |
| B4 | 2100 | 100.00 | 0.974 | 0.993 | 7 / 7 |
| B4 | 2101 | 14.17 | 0.996 | 0.999 | 1 / 7 |
| B4 | 2102 | 99.21 | 0.986 | 0.993 | 7 / 7 |

## Alignment and consistency diagnostics

First/final epoch losses below are raw, unweighted means over source steps.

| Arm | First MMD² | Final MMD² | First KL | Final KL |
| --- | ---: | ---: | ---: | ---: |
| B1 | 0.123986 | 0.012504 | 0.000000 | 0.000000 |
| B2 | 0.124739 | 0.020930 | 0.000000 | 0.000000 |
| B3 | 0.000000 | 0.000000 | 0.000133 | 0.000807 |
| B4 | 0.124733 | 0.015578 | 0.000021 | 0.002121 |

Inspect `summary.json` in each run for per-class recall, prediction counts,
spatial attention maps, slot cosine, and attention-map cosine.

## Conclusion

B0 has the highest mean fixed-epoch target OA. B2 does not beat B1 on any seed,
and B4 only modestly improves B2 while remaining well below B0. B3 is below B0
on every seed. B1, B2, and B4 collapse to one predicted class on seed 2101,
with source validation OA near chance. B3 fits source validation but transfers
poorly on two seeds. These observations do not support the proposed slot-wise
alignment and target-consistency gains under the locked V1 settings; they do
not isolate the cause of instability or rule out other implementations.
