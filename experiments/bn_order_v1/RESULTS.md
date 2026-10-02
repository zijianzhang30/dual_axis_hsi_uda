# BN update-order causal control

Seeds 2100/2101/2102; original BiDA tokenizer, depth 3, dropout 0.1,
source CE, fixed epoch 200. Collapse threshold: >=95% class-6 predictions.

| Arm | OA mean ± sample SD | AA mean ± sample SD | Class-6 collapse |
| --- | ---: | ---: | ---: |
| Source then target | 73.18 ± 8.21 | 54.05 ± 18.46 | 1/3 |
| Target then source | 71.99 ± 8.42 | 48.48 ± 20.94 | 1/3 |
| Joint-Batch / Fixed-Mixture V1 | 79.50 ± 1.46 | 71.90 ± 1.72 | 0/3 |

| Seed | Original OA | Reverse OA | Order OA change | Order AA change | Original class-6 share | Reverse class-6 share | Joint OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 78.41 | 77.44 | -0.97 | -4.16 | 63.66% | 68.50% | 78.55 |
| 2101 | 63.72 | 62.30 | -1.42 | -8.43 | 95.93% | 98.09% | 78.77 |
| 2102 | 77.40 | 76.24 | -1.17 | -4.10 | 72.88% | 77.38% | 81.19 |

## Mechanistic interpretation

Changing only running-buffer update order lowers OA in every seed by
0.97, 1.42 and 1.17 points (mean -1.19), and lowers mean AA by 5.57
points. It leaves the same seed-2101 class-6 collapse in place, increasing
that class's prediction share from 95.93% to 98.09%. In seed 2102,
class-7 recall falls from 0.73% to zero. The order effect is therefore real
and affects minority coverage, but merely reversing order does not
explain or reproduce Joint-Batch's stability improvement.

This supports studying the full shared-training-statistics protocol.
It does not prove that order bias is irrelevant or that target gradients
alone drive the gain. Joint-Batch and Fixed-Mixture V1 are the same arm
under equal batch sizes and matched EMA rate; their equivalence cannot
separate training moments from buffer moments or target gradient paths.
A future buffer-only balanced update preserving domain-specific training
normalization, followed by a detached-target-moment shared-normalization
control, would isolate these contributions. Neither is run or selected
in this study.

## What is isolated

Reverse-order source CE matches every epoch of the original control exactly
in all three seeds. All final non-BN-buffer tensors also match exactly.
Target/source stem order changes; source labels, downstream branches and
dropout consumption retain their original order. This comparison therefore
isolates the effect of BN running-buffer update order on inference.

Sequential source-then-target fresh coefficients are 0.09/0.10; reverse
coefficients are 0.10/0.09, with old-state weight 0.81 in both cases.
Fresh normalized domain weights are 47.37/52.63 versus 52.63/47.37.

## Joint-Batch identity

Fixed-Mixture V1 already calls native BN once on concatenated equal-sized
domain activations in each stem layer with momentum 0.19. An independently
implemented concatenated stem produces identical tokens (max error 0),
source/target input gradients, parameter gradients and BN buffers on the
verification fixture. It is the Joint-Batch arm under the same EMA rate.
The existing three-seed frozen results are reused; there is no distinct
special mixture implementation to compare against Joint-Batch.

These comparisons cannot isolate pooled training normalization from
the between-domain variance term or target moment gradients. A buffer-only
balanced-mixture control or detached-target-moment training intervention
would be needed to distinguish those mechanisms, with a protocol frozen
before further target outcomes.

## Frozen reverse trajectories

| Seed | Class-6 collapsed grid points | First collapsed epoch |
| --- | ---: | ---: |
| 2100 | 0/20 | none |
| 2101 | 17/20 | 10 |
| 2102 | 0/20 | none |

Verified all 120 new checkpoint/prediction SHA256 records and three
frozen manifests. All twenty predictions per seed were frozen before target
metrics. Full per-class recalls, prediction shares, buffer differences,
and trajectories are in results/bn_order_v1/summary.json.
