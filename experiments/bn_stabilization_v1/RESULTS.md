# BN stabilization V1: three-seed result

Houston13 to Houston18, original BiDA tokenizer, depth 3, dropout 0.1,
source self CE, fixed epoch 200; seeds 2100/2101/2102. Collapse means
at least 95% of target predictions are class 6. Mean plus sample SD.

| BN arm | Collapsed seeds | OA | AA | Kappa |
| --- | ---: | ---: | ---: | ---: |
| Original Mixed BN | 1/3 | 73.18 ± 8.21 | 54.05 ± 18.46 | 0.440 ± 0.280 |
| DSBN | 0/3 | 62.44 ± 5.68 | 73.43 ± 1.55 | 0.483 ± 0.056 |
| Fixed-Mixture 0.5/0.5 | 0/3 | 79.50 ± 1.46 | 71.90 ± 1.72 | 0.665 ± 0.024 |

Fixed-Mixture passes the preregistered confirmation gate: True.
It preserves the previously successful seeds and repairs seed 2101. The
three-seed result supports a controlled follow-up of shared mixed training
normalization. DSBN restores balanced recall but does not preserve OA.

The intervention does not isolate update order: Fixed-Mixture also uses
pooled current moments for both domains, includes between-domain mean
variance, and allows target gradients through those moments. Its EMA rate
matches the original per-step forgetting rate. This is evidence for the
complete normalization protocol, not attribution to a single component.

Minority-class instability remains: Fixed-Mixture class-1 and class-7 recall
sample SDs are about 24.50 and 22.66 points. Three optimization seeds in one
transfer pair do not establish stability across datasets or class priors.

## Per-seed fixed endpoints

| Arm | Seed | OA | AA | Class-6 share | Recall coverage | Source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Mixed BN | 2100 | 78.41 | 66.32 | 63.66% | 7/7 | 79.53 |
| Original Mixed BN | 2101 | 63.72 | 32.82 | 95.93% | 5/7 | 74.02 |
| Original Mixed BN | 2102 | 77.40 | 63.01 | 72.88% | 7/7 | 67.72 |
| DSBN | 2100 | 63.65 | 74.72 | 36.57% | 7/7 | 100.00 |
| DSBN | 2101 | 56.26 | 71.71 | 30.55% | 7/7 | 100.00 |
| DSBN | 2102 | 67.43 | 73.85 | 42.00% | 7/7 | 100.00 |
| Fixed-Mixture 0.5/0.5 | 2100 | 78.55 | 70.03 | 52.02% | 7/7 | 100.00 |
| Fixed-Mixture 0.5/0.5 | 2101 | 78.77 | 72.25 | 61.46% | 7/7 | 100.00 |
| Fixed-Mixture 0.5/0.5 | 2102 | 81.19 | 73.42 | 58.16% | 7/7 | 100.00 |

## Predeclared confirmation gate

| Criterion | DSBN | Fixed-Mixture |
| --- | --- | --- |
| no_epoch200_class6_collapse | True | True |
| mean_oa_not_lower | False | True |
| mean_aa_not_lower | True | True |
| oa_sample_sd_lower | True | True |
| successful_control_seeds_mean_oa_drop_at_most_2pp | False | True |

DSBN passes all criteria: **False**. Mean OA change on previously noncollapsed seeds 2100/2102: -12.37 points.

Fixed-Mixture 0.5/0.5 passes all criteria: **True**. Mean OA change on previously noncollapsed seeds 2100/2102: +1.96 points.

## Per-class recall

| Arm | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Mixed BN | 33.90 ± 26.69 | 50.67 ± 43.90 | 45.71 ± 14.91 | 81.82 ± 0.00 | 63.47 ± 42.34 | 94.93 ± 4.51 | 7.86 ± 12.98 |
| DSBN | 91.50 ± 4.79 | 58.34 ± 8.55 | 70.45 ± 4.64 | 98.48 ± 2.62 | 71.31 ± 2.92 | 59.24 ± 9.19 | 64.68 ± 7.31 |
| Fixed-Mixture 0.5/0.5 | 58.96 ± 24.50 | 76.49 ± 3.08 | 58.25 ± 6.67 | 83.33 ± 2.62 | 94.45 ± 2.08 | 86.93 ± 3.85 | 44.89 ± 22.66 |

## Frozen trajectory stability

| Arm | Seed | Collapsed grid points | First collapsed epoch |
| --- | ---: | ---: | ---: |
| DSBN | 2100 | 0/20 | none |
| DSBN | 2101 | 0/20 | none |
| DSBN | 2102 | 0/20 | none |
| Fixed-Mixture 0.5/0.5 | 2100 | 0/20 | none |
| Fixed-Mixture 0.5/0.5 | 2101 | 0/20 | none |
| Fixed-Mixture 0.5/0.5 | 2102 | 0/20 | none |

Avoiding class-6 collapse alone is not sufficient to establish overall
trajectory stability. DSBN seed 2101 reaches only 12.88% OA at epoch 180,
with 51.51% of predictions assigned to class 4; DSBN seed 2102 reaches
21.25% OA at epoch 20. Fixed-Mixture has no comparable later breakdown
in the frozen grid. These observations do not change the primary endpoint
or predeclared gate.

## Causal and reproducibility checks

- DSBN seed 2100: final learned tensors exactly equal control = True; maximum epoch source-CE replay error = 0.
- DSBN seed 2101: final learned tensors exactly equal control = True; maximum epoch source-CE replay error = 0.
- DSBN seed 2102: final learned tensors exactly equal control = True; maximum epoch source-CE replay error = 0.

DSBN changes only domain buffers. Fixed-Mixture uses shared current batch
moments and changes gradients/learned weights. The fixed-mixture EMA rate
is 0.19 per paired step, matching two original momentum-0.1 updates'
forgetting rate. All arms have 376,567 trainable parameters.

Verified 240 new checkpoint/prediction SHA256 records plus six manifests.
Target predictions for all grid checkpoints were frozen before post-hoc
target metrics. Original controls have only frozen epoch-200 endpoints.

Artifacts and summary.json are under results/bn_stabilization_v1/.
Trajectory figure: results/bn_stabilization_v1/trajectories.png.
