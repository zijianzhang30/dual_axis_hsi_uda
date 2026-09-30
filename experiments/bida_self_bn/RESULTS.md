# BiDA-self BN decomposition: seed 2100

The [protocol](PROTOCOL.md) was frozen before training. The intervention
replayed the completed BiDA-self training exactly while preventing target BN
updates from persisting beyond each step. Both arms use fixed epoch-200 target
self inference.

| BN buffers | Target OA | AA | Kappa | Source-val OA | Largest prediction share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source-only BN | 61.55% | 15.20% | 0.024 | 100.00% | 99.27% |
| Mixed source/target BN | **78.41%** | **66.32%** | **0.627** | 79.53% | 63.66% |

Mixed BN improves OA by **16.86 points**, AA by **51.12 points**, and Kappa by
0.603. Source-only BN predicts class 6 for 52,516 of 52,901 target pixels and
has nonzero recall for only four classes. Its per-class recalls are
`[0.00, 0.57, 0.65, 0.00, 5.30, 99.91, 0.00]` percent. This is a clear
majority-class collapse.

## Causal control

- Every epoch's source CE is exactly equal to the completed mixed-BN
  BiDA-self history; maximum absolute difference is `0.0`.
- All 50 non-BN state tensors in the final source-only and mixed-BN
  checkpoints are bitwise equal. Only the six BN running-statistic/count
  tensors differ.
- Target batches were still forwarded and consumed the same dropout RNG. Their
  BN buffers were restored after backward and therefore never reached the next
  step or final checkpoint.
- The source-only intervention reaches 100% source-val OA, while mixed BN has
  79.53%. Source-domain validation therefore prefers the model that collapses
  on target and cannot select the useful BN state.

For seed 2100, target-conditioned mixed BN is not a small contributor to
BiDA-self. It is necessary for the observed cross-scene class coverage under
this training path. The result also explains why BiDA-self can outperform A1
despite modest source validation: its inference statistics deliberately move
away from the source-only optimum.

This does not yet show that transformer depth is unimportant. It shows that a
depth ablation must retain mixed BN; otherwise a collapse could be incorrectly
attributed to shallower token refinement. No BN momentum or update-order
search was performed.

## Frozen artifacts

Machine-readable metrics, history, confusion matrices, checkpoint, and frozen
predictions are under `results/bida_self_bn/diagnostic_2100_formal/`.

- Frozen manifest SHA256:
  `0371ee01189d2327bde40a333ecbdf85fa2d8f9c8740f3b9069d5b7e28fb5ff5`
- Final checkpoint SHA256:
  `bccf9d1ed36065c60b9ef0ff490281a9deffacc3c9cf5a4015e74fb995f57807`
- Prediction SHA256:
  `117789010c60995d9574db86eabfb0da744e4fce4da6e920e6d46d123b7e1ab4`

Training took 140 seconds and target inference 2.37 seconds on GPU 4. The
model has 376,567 trainable parameters. This is one diagnostic seed and does
not establish multi-seed stability.
