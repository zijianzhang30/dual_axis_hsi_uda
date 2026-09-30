# H2 BiDA-self decomposition: seed 2100

The [protocol](PROTOCOL.md) was frozen before training. BiDA-self retained the
released BiDA architecture, three self-attention blocks, dropout, and shared
source/target BatchNorm updates. Its only loss was source self CE. Coupled
distillation, EMA consistency, and MMD were disabled. The primary result is
target self inference at fixed epoch 200.

| Model | Target OA | AA | Kappa | Source-val OA at epoch 200 |
| --- | ---: | ---: | ---: | ---: |
| A1-GN, existing seed-2100 reference | 73.58% | 62.13% | 0.491 | 100.00% |
| **BiDA-self, new diagnostic** | **78.41%** | **66.32%** | **0.627** | 79.53% |
| Strict BiDA, existing seed-2100 reference | 78.75% | 66.95% | 0.617 | 81.10% |

BiDA-self is +4.83 OA and +4.19 AA points above A1-GN. Strict BiDA is only
+0.34 OA and +0.62 AA points above BiDA-self. BiDA-self lies within the
predeclared two-point equivalence region around Strict BiDA, while the
predeclared condition for a material Strict-BiDA objective advantage fails.

For this seed, the evidence supports the self representation, target-conditioned
BN, and regularization as the dominant explanation of BiDA's fixed-epoch
advantage. It does not support coupled distillation, EMA consistency, and MMD
as the main source of that advantage.

## Target diagnostics

BiDA-self per-class recalls are
`[40.58, 77.24, 57.81, 81.82, 93.73, 90.26, 22.84]` percent. All seven
classes retain nonzero recall. Its prediction shares are
`[1.43, 8.11, 4.77, 0.10, 18.57, 63.66, 3.36]` percent, so it does not show
the one-class collapse observed in earlier MMD and full-source-set experiments.
Mean target entropy is 0.1449.

The initial model-state SHA256 exactly matches the existing Strict BiDA
seed-2100 run:
`dadb50a683a2bab506df66c317cbea8d08aee955bab5281013dd4604d6fa40b5`.
BiDA-self has 376,567 trainable parameters. Training took 147 seconds and
target inference took 2.47 seconds on GPU 4 in this run.

## Interpretation limits

This is one optimization seed and one fixed source split. Removing the EMA
forward and auxiliary losses also changes the training RNG consumption after
initialization, so BiDA-self and Strict BiDA do not share an identical weight
trajectory. Their initialization, data protocol, optimizer, architecture, and
fixed endpoint are matched; the comparison estimates the whole auxiliary
objective package rather than an exact per-step intervention.

The result also does not isolate tokenization, transformer depth, dropout, and
BN from one another. H1 and the A1-BN experiment already establish that BN
statistics materially affect predictions, while H2 shows that the complete
BiDA self path is sufficient to reproduce nearly all of Strict BiDA's
seed-2100 endpoint. Further work should decompose the self representation
before designing stronger source-target interaction.

## Frozen artifacts

Machine-readable metrics and confusion matrix are in
`results/bida_self/diagnostic_2100/results.json`. The prediction artifact was
frozen before target labels were collected.

- Frozen manifest SHA256:
  `6909e5530a53c33a817cc8f55e5923623a2ab726a6af92782533d98ccc72c144`
- Final checkpoint SHA256:
  `963e4c12798b120484db84f4c879a71fb444073bfde279e45b5b4bf66516fefe`
- Prediction SHA256:
  `abc65827ef46b0486e90fe438ebb18b41e5cccac16449c1e5e04b659cf6cf14c`

The released Houston18 GT mask still defines target training centers. Target
class values entered neither training nor checkpoint selection.
