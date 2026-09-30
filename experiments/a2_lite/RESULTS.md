# A1 vs A2-lite: explicit global spectral tokens

Houston13 → Houston18, three optimization seeds, fixed epoch 200.
Both arms use the same Strict BiDA data/training pipeline and source CE only.
The released target GT mask defines eligible target centers; target labels
enter neither loss nor checkpoint selection.

| Arm | Target OA (%) | Target AA (%) | Kappa | Parameters |
| --- | ---: | ---: | ---: | ---: |
| a1 | 72.10 ± 1.39 | 60.73 ± 1.34 | 0.47 ± 0.02 | 405,255 |
| a2_lite | 68.45 ± 7.70 | 59.09 ± 0.43 | 0.44 ± 0.06 | 546,823 |

Paired OA difference A2-lite − A1: **-3.65 ± 6.52 points**.

## Per-seed outcome

Spectral class-query mass is the summed attention weight over four spectral
tokens; it is a usage diagnostic, not a causal attribution.

| Seed | A1 OA | A2-lite OA | Difference | A1 AA | A2-lite AA | A1 source val OA | A2-lite source val OA | Spectral attention mass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 73.58 | 73.99 | +0.41 | 62.13 | 59.42 | 100.00 | 100.00 | 0.328 |
| 2101 | 70.82 | 59.65 | -11.17 | 59.46 | 58.61 | 100.00 | 100.00 | 0.356 |
| 2102 | 71.90 | 71.71 | -0.19 | 60.59 | 59.24 | 100.00 | 100.00 | 0.380 |

## Per-class target recall (%)

Mean ± sample SD across seeds; classes follow the BiDA loader order.

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | 68.37 ± 10.14 | 46.01 ± 14.60 | 56.45 ± 3.37 | 100.00 ± 0.00 | 60.40 ± 1.53 | 93.45 ± 0.76 | 0.39 ± 0.68 |
| a2_lite | 60.95 ± 6.96 | 60.91 ± 4.81 | 53.77 ± 0.19 | 100.00 ± 0.00 | 50.47 ± 3.02 | 87.46 ± 13.65 | 0.07 ± 0.07 |

## Spectral attention over latent wavelength positions

Rows are spectral query slots; columns are the 12 latent spectral positions.
Entries are average target attention weights, rounded to three decimals.

### Seed 2100

| Query | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.086 | 0.096 | 0.088 | 0.076 | 0.071 | 0.075 | 0.082 | 0.090 | 0.092 | 0.090 | 0.080 | 0.075 |
| 2 | 0.082 | 0.094 | 0.086 | 0.081 | 0.081 | 0.088 | 0.095 | 0.094 | 0.085 | 0.071 | 0.073 | 0.070 |
| 3 | 0.059 | 0.073 | 0.093 | 0.095 | 0.085 | 0.072 | 0.069 | 0.081 | 0.092 | 0.096 | 0.090 | 0.093 |
| 4 | 0.080 | 0.078 | 0.085 | 0.085 | 0.086 | 0.084 | 0.084 | 0.086 | 0.086 | 0.082 | 0.076 | 0.088 |

### Seed 2101

| Query | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.084 | 0.068 | 0.065 | 0.062 | 0.061 | 0.060 | 0.068 | 0.081 | 0.095 | 0.111 | 0.126 | 0.120 |
| 2 | 0.091 | 0.089 | 0.087 | 0.080 | 0.080 | 0.085 | 0.088 | 0.090 | 0.085 | 0.081 | 0.072 | 0.070 |
| 3 | 0.068 | 0.075 | 0.088 | 0.091 | 0.087 | 0.077 | 0.078 | 0.088 | 0.092 | 0.086 | 0.086 | 0.084 |
| 4 | 0.072 | 0.067 | 0.070 | 0.074 | 0.078 | 0.084 | 0.088 | 0.096 | 0.099 | 0.092 | 0.096 | 0.086 |

### Seed 2102

| Query | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.097 | 0.093 | 0.094 | 0.089 | 0.084 | 0.078 | 0.079 | 0.082 | 0.080 | 0.075 | 0.070 | 0.078 |
| 2 | 0.104 | 0.083 | 0.100 | 0.110 | 0.101 | 0.079 | 0.062 | 0.059 | 0.060 | 0.069 | 0.088 | 0.086 |
| 3 | 0.093 | 0.085 | 0.076 | 0.074 | 0.078 | 0.086 | 0.091 | 0.092 | 0.090 | 0.087 | 0.075 | 0.073 |
| 4 | 0.116 | 0.101 | 0.092 | 0.083 | 0.075 | 0.068 | 0.064 | 0.068 | 0.074 | 0.086 | 0.087 | 0.086 |

The attention distribution alone cannot establish which spectral information
caused a prediction. Per-run summaries retain full confusion matrices and
mean spatial/spectral attention maps.
