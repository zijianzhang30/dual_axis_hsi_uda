# H1 BatchNorm Diagnostic Results

The protocol was frozen in [PROTOCOL.md](PROTOCOL.md) before this run. This
diagnostic used the unmodified Strict BiDA seed-2100 epoch-200 checkpoint. It
performed no optimization and did not modify A1.

## Frozen result

| BN buffers | Target OA (%) | AA (%) | Kappa | Nonzero-recall classes | Largest prediction share | Mean entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original mixed BN | **78.75** | **66.95** | **0.617** | 7/7 | 68.52% | 0.0924 |
| Source-recalibrated BN | 63.51 | 18.31 | 0.096 | 3/7 | 97.40% | 0.0371 |
| Target-recalibrated BN | 66.71 | 65.13 | 0.523 | 7/7 | 40.91% | 0.0669 |

Target-only relative to source-only recalibration:

- OA: **+3.20 percentage points**;
- AA: **+46.81 points**;
- nonzero-recall classes: **7/7** versus **3/7**;
- largest predicted-class share: **−56.48 points**.

All four predeclared conditions passed. H1 therefore passes its diagnostic
gate, and H2 was not started.

## Mechanism diagnostics

Source-only recalibration produced a majority-class failure: 51,523 of 52,901
pixels (97.40%) were predicted as class 6. Target-only recalibration removed
that collapse and restored nonzero recall for every class. Its per-class
recalls were `[78.42, 52.97, 74.76, 45.45, 59.12, 66.55, 78.60]` percent.

Changing BN statistics alone changed 56.86% of predictions between the source
and target variants. The original mixed checkpoint disagreed with source-only
and target-only variants on 28.97% and 32.89% of pixels, respectively. The
second BN layer showed a large source-target running-mean L2 distance of 3.77.
BN statistics therefore have a material causal effect on target predictions.

The original mixed BN remains substantially better than target-only BN:
78.75% versus 66.71% OA. The result supports target-conditioned normalization
as an important adaptation channel, but does not support replacing the learned
mixed statistics with target-only recalibration. The original checkpoint's BN
buffers arise throughout joint source/target training and interact with the
learned weights; post-hoc target recalibration is a different intervention.

## Frozen artifacts

- Machine-readable metrics: `results/bn_diagnostic/h1_seed2100_formal/results.json`
- Frozen manifest SHA256: `40a2ad85da67e9308683627bc25fcb1ebb7f24560105462a731dac516cf5b03d`
- Original prediction SHA256: `2a88a4967158ffbb26a15cfc59a55098a36ca071c5eab0067cf30288c1283dec`
- Source-BN prediction SHA256: `8198274c307b2895a3d7e371cc828da5c16f135c8641a27e962ec3e7d7db8fdc`
- Target-BN prediction SHA256: `22edb848f5605434c7c6761f2864039538dd893456f31246becd14a783d315b1`

This is one diagnostic seed. It establishes a mechanism signal under the
predeclared gate, not a multi-seed performance claim.
