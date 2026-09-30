# A1 BatchNorm transplant: seed 2100

The [protocol](PROTOCOL.md) was frozen before training. Both arms used one
shared weight trajectory with source CE. N0 updated BN from source batches;
N1 updated BN from the same source batches and paired unlabeled target batches.
Evaluation used the target self branch at fixed epoch 200.

| Arm | Target OA | AA | Kappa | Source-val OA | Largest predicted class share |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0: A1-BN, source statistics | 73.94% | 50.72% | 0.492 | 100% | 76.22% |
| N1: A1-BN, source + target statistics | 75.28% | 56.85% | 0.503 | 100% | 79.53% |

**N1 − N0:** +1.34 OA points and +6.13 AA points. The two target prediction
sets disagree on 6.65% of the 52,901 pixels. N1 improves class 1 recall from
16.70% to 52.03% and class 3 from 32.89% to 49.22%, but reduces class 2
from 61.03% to 53.87%. Class 7 remains nearly absent: 0.16% and 0.24%
recall in N0 and N1, respectively. Class 6 receives 79.53% of N1 predictions,
above its true 60.99% target share. Thus neither arm solves class-prior bias.

The existing A1-GN paired seed-2100 reference has 73.58% OA and 62.13% AA.
N1 is +1.70 OA points but −5.28 AA points against that separate run. This
comparison is contextual: the only exact paired contrast here is N1 versus N0.

## Causal control and gate

- The maximum source-logit difference across all 3,600 paired steps was
  exactly `0.0`.
- All 48 non-BN-buffer state tensors in the two final checkpoints are exactly
  equal. Only the nine BN running-statistic/count tensors differ.
- Shared trainable-parameter SHA256:
  `ed5d9f61473fe1604748e45ed50471ac398b051ea440e0ce0c4df3168213667d`.
- Training the shared trajectory took 627 seconds on GPU 4. Target inference
  took 9.8 seconds for N0 and 10.1 seconds for N1. Both have 405,255
  trainable parameters and identical target-only inference cost.

The predeclared gate **failed** because the OA gain was +1.34 rather than at
least +2.0 points. AA, class coverage, and largest-class-share conditions
passed. This single seed shows that target BN updates can change A1 target
predictions and improve OA/AA here; it does not establish a stable backbone
gain. The present protocol therefore stops without expanding to three seeds
or changing BN hyperparameters.

## Frozen artifacts

Machine-readable metrics, confusion matrices, per-class recalls, history,
checkpoints, and predictions are under
`results/a1_bn/diagnostic_2100/`. Predictions and SHA256 hashes were frozen
before target labels were collected. Frozen manifest SHA256:
`77811d63e9e2dea0179e61e1c1245b66e9c8166ae0f401a6abe5fc5d46123474`.

The target training centers follow the released Houston18 GT mask. Target
class values were ignored by training and checkpoint selection, but the mask
remains a protocol limitation.
