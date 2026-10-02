# Tokenizer x readout factorial: stress seed2101

All cells use Full Joint BN and zero BiDA blocks. Single-seed diagnosis,
not a three-seed stability estimate. query_class is reused; three cells newly trained.

## Fixed epoch200 primary

| Tokenizer | Readout | OA | AA | Kappa | Class6 share | Class7 recall | Recall coverage | Source-val OA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bida | mean | 64.46 | 72.17 | 0.505 | 36.06% | 70.44% | 7/7 | 100.00 |
| bida | class | 75.81 | 73.12 | 0.624 | 51.30% | 50.83% | 7/7 | 100.00 |
| query | mean | 70.14 | 42.28 | 0.388 | 81.14% | 0.00% | 6/7 | 100.00 |
| query | class | 74.55 | 47.24 | 0.467 | 83.41% | 0.00% | 6/7 | 100.00 |

## Factorial contrasts (percentage points)

| Contrast | OA | AA |
| --- | ---: | ---: |
| query_minus_bida_under_class | -1.26 | -25.88 |
| query_minus_bida_under_mean | +5.68 | -29.89 |
| class_minus_mean_under_query | +4.41 | +4.96 |
| class_minus_mean_under_bida | +11.35 | +0.95 |
| interaction | -6.94 | +4.01 |

Interaction = (Class minus Mean under Query) minus (Class minus Mean under BiDA).
These are optimization-seed-specific descriptive differences, not confidence intervals.

## Oracle diagnostic only

| Arm | Best epoch | OA | Associated AA |
| --- | ---: | ---: | ---: |
| bida_mean | 50 | 72.45 | 64.89 |
| bida_class | 40 | 76.50 | 66.79 |
| query_mean | 180 | 71.30 | 35.59 |
| query_class | 110 | 75.87 | 51.94 |

## Fixed per-class recall

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bida_mean | 65.11 | 60.56 | 65.12 | 90.91 | 94.16 | 58.87 | 70.44 |
| bida_class | 74.21 | 68.55 | 65.09 | 81.82 | 91.13 | 80.24 | 50.83 |
| query_mean | 22.03 | 8.12 | 28.07 | 59.09 | 81.72 | 96.92 | 0.00 |
| query_class | 10.13 | 62.87 | 50.13 | 54.55 | 53.92 | 99.08 | 0.00 |

## Fixed prediction distribution

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bida_mean | 1394 (2.64%) | 3443 (6.51%) | 3907 (7.39%) | 70 (0.13%) | 13540 (25.59%) | 19078 (36.06%) | 11469 (21.68%) |
| bida_class | 1557 (2.94%) | 4082 (7.72%) | 3582 (6.77%) | 18 (0.03%) | 11472 (21.69%) | 27137 (51.30%) | 5053 (9.55%) |
| query_mean | 503 (0.95%) | 400 (0.76%) | 823 (1.56%) | 13 (0.02%) | 8239 (15.57%) | 42923 (81.14%) | 0 (0.00%) |
| query_class | 176 (0.33%) | 3336 (6.31%) | 1894 (3.58%) | 12 (0.02%) | 3356 (6.34%) | 44127 (83.41%) | 0 (0.00%) |

## Fixed angular probes

| Arm | Mode | Domain | Token concentration | Pre-LN concentration | z norm | z concentration | Cos(z,w6) | Margin6 | Class6 share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bida_mean | eval | source | 0.744 | 0.744 | 8.218 | 0.212 | 0.059 | -6.170 | 15.92% |
| bida_mean | eval | target | 0.872 | 0.872 | 8.209 | 0.510 | 0.093 | -2.711 | 37.01% |
| bida_mean | joint_train | source | 0.742 | 0.742 | 8.218 | 0.195 | 0.040 | -6.384 | 15.92% |
| bida_mean | joint_train | target | 0.871 | 0.871 | 8.210 | 0.507 | 0.082 | -2.873 | 36.82% |
| bida_class | eval | source | 0.729 | 0.182 | 8.139 | 0.180 | 0.033 | -6.933 | 15.92% |
| bida_class | eval | target | 0.847 | 0.502 | 8.144 | 0.504 | 0.303 | -0.032 | 51.76% |
| bida_class | joint_train | source | 0.730 | 0.163 | 8.138 | 0.160 | 0.013 | -7.233 | 15.92% |
| bida_class | joint_train | target | 0.847 | 0.494 | 8.144 | 0.496 | 0.277 | -0.369 | 50.24% |
| query_mean | eval | source | 0.135 | 0.135 | 8.114 | 0.133 | 0.051 | -6.552 | 15.92% |
| query_mean | eval | target | 0.883 | 0.883 | 8.128 | 0.885 | 0.499 | 4.409 | 82.32% |
| query_mean | joint_train | source | 0.130 | 0.130 | 8.114 | 0.127 | 0.042 | -6.665 | 15.92% |
| query_mean | joint_train | target | 0.883 | 0.883 | 8.128 | 0.885 | 0.496 | 4.332 | 81.49% |
| query_class | eval | source | 0.371 | 0.133 | 8.106 | 0.130 | 0.024 | -7.027 | 15.92% |
| query_class | eval | target | 0.861 | 0.867 | 8.114 | 0.867 | 0.636 | 4.757 | 83.06% |
| query_class | joint_train | source | 0.369 | 0.130 | 8.106 | 0.126 | 0.026 | -7.013 | 15.92% |
| query_class | joint_train | target | 0.861 | 0.867 | 8.114 | 0.868 | 0.635 | 4.738 | 83.54% |

## Trajectory collapse and parameters

| Arm | Class6 collapse grid points | Any-class ≥95% grid points | Parameters |
| --- | ---: | ---: | ---: |
| bida_mean | 0/20 | 0/20 | 222,455 |
| bida_class | 0/20 | 0/20 | 256,119 |
| query_mean | 0/20 | 0/20 | 256,055 |
| query_class | 0/20 | 0/20 | 289,719 |

## Interpretation of this stress-seed factorial

The same Class Query readout is compatible with balanced target coverage when fed BiDA tokens: AA73.12%, class6 share51.30%, class7 recall50.83%. With Query tokens the reused cell has AA47.24%, class6 share83.41%, class7 recall0%. This argues against Class Query being a tokenizer-independent sole cause.
Replacing Class Query with mean pooling does not repair the Query cell: OA falls4.41pp and AA falls4.96pp, with class7 recall still0%. Query z direction concentration is0.885 under Mean versus0.867 under Class Query.
The completed BiDA+Mean cell retains endpoint AA72.17% and seven-class recall, with z concentration0.510 versus0.504 under BiDA+Class Query. Both Query cells have much higher endpoint target concentration than both BiDA cells. The reader-independent adverse class-balanced endpoint signature therefore tracks the Query tokenizer package under this Full Joint setting.
Do not label BiDA+Mean globally stable: at epoch170 target OA drops to14.03%, AA40.91%, with64.37% predictions in class4 and zero class6 predictions, before recovering. Its source CE spikes and source-val OA is70.87% at that checkpoint. A class6-specific collapse threshold misses this alternative transient failure.
OA alone gives a misleading tokenizer ranking under mean pooling: Query exceeds BiDA by5.68pp OA but loses29.89pp AA. BiDA+Mean class6 recall is58.87%, versus80.24% with Class Query; the dominant target class heavily influences OA. Class Query improves BiDA OA by11.35pp and AA by0.95pp rather than causing collapse.
Current joint-training moments reproduce the high Query concentration/cosine signatures, so saved inference buffers alone do not explain this pattern. Stage probes are observational and use the same unlabeled sample in all cells.
Provisional diagnosis: prioritize the spatial Query package and its learned domain representation, not removal of Class Query or extra losses. This does not isolate learnable queries from positional encoding, attention, LayerNorm, FFN or parameter count, and does not prove Query is intrinsically flawed. All learned components co-adapt; one stress seed is insufficient for a general attribution.

## Scope and checks

Verified 160 checkpoint/prediction SHA256 records and four manifests. All new predictions frozen before post-hoc GT metrics.
Common pipeline initialization matches the reused control; cell selection preserves RNG state. Retained stem/classifier/head tensors exactly match at initialization.
BiDA/Query is a tokenizer package comparison; Class Query/Mean changes capacity and nonlinear readout. All trainable state co-adapts. One stress seed selected after prior outcomes cannot establish general causality or stability.
Fixed samples and source/target probes are shared. Eval uses saved buffers; joint-train probe uses current moments with dropout disabled and restores/hash-checks state. Not live historical activations.
No added loss, BN change, residual scale, ratio or oracle tuning. The main reference remains Full Joint BN + BiDA tokenizer with its original depth3: 79.50±1.46% over three seeds.
