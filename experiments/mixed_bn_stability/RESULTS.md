# Mixed-BN seed-instability diagnostic

Six frozen epoch-200 models were audited: the BiDA-tokenizer controls and
query-tokenizer Hybrids for seeds 2100/2101/2102. Each model was evaluated with
its original mixed BN buffers, deterministic source-only cumulative BN
recalibration, and unlabeled target-only cumulative BN recalibration. All
trainable tensors, including stem, tokenizer, transformer, and classifier,
were held byte-identical within each comparison.

## Main finding

The collapse is strongly controlled by BN buffers, but target-only cumulative
recalibration is not itself an OA solution. In the collapsed Hybrid seed 2101,
changing only BN buffers produces:

| BN buffers | Target OA | AA | Class-6 prediction share |
| --- | ---: | ---: | ---: |
| Original mixed | 62.59% | 26.38% | 98.19% |
| Source recalibrated | 60.99% | 14.29% | 100.00% |
| Target recalibrated | 54.63% | **69.82%** | 27.56% |

Target recalibration changes 70.98% of seed-2101 Hybrid predictions and removes
the class-6 collapse without changing learned weights. AA rises by 43.44
points and all-class coverage is restored. This directly demonstrates a large
BN-buffer/classifier representation mismatch. OA falls because target-only
cumulative statistics overcorrect toward minority-class predictions; they are
a causal probe rather than a ready checkpoint rule.

The same qualitative behavior occurs in all six models. Source-only
recalibration sends 98.76% to 100% of predictions to class 6. Target-only
recalibration reduces class-6 share to 27.56%--37.11% and raises AA to
69.82%--75.68%, but lowers OA to 54.63%--65.02%. Original mixed BN occupies a
high-OA intermediate regime when training is stable. The stability problem is
therefore not simply whether target statistics are present; it is how the
source/target running-statistic mixture evolves and aligns with learned
weights.

For collapsed seed 2101, original mixed predictions are functionally close to
source-recalibrated predictions: they disagree on only 4.07% of target pixels
for the control and 1.81% for the Hybrid. In contrast, original and target-
recalibrated predictions disagree on 66.23% and 70.98%, respectively. This is
the clearest evidence that the failed mixed-BN state behaves like the
source-collapse regime at inference.

## Target trajectory

Hybrid seed 2101 collapses early and oscillates rather than degrading smoothly:

| Epoch | Target OA | AA | Class-6 share |
| --- | ---: | ---: | ---: |
| 10 | 76.79% | 55.19% | 71.20% |
| 20 | 64.02% | 33.00% | 95.90% |
| 110 | 61.60% | 23.83% | 99.30% |
| 120 | 74.37% | 56.53% | 80.31% |
| 180 | 62.34% | 26.44% | 98.40% |
| 200 | 62.59% | 26.38% | 98.19% |

The principal collapse occurs between epochs 10 and 20. A large temporary
recovery appears at epoch 120, followed by renewed collapse. Consecutive BN
buffer movement alone does not explain these events: across epochs 20--200,
the Pearson correlations between class-6 share and consecutive running-mean or
log-variance change magnitude are only -0.15 and -0.15 for seed 2101. Large BN
updates also occur in successful seeds without collapse. The failure depends
on compatibility among current learned features, classifier weights, and BN
buffers rather than a scalar amount of BN drift.

Source validation does not reveal this trajectory. Hybrid seed 2101 ends with
74.80% source-val OA while 98.19% of target predictions are class 6. Target
coverage can therefore change radically while source discrimination remains
usable.

## Cross-seed final BN distances

Both BN layers differ substantially across seeds. For example, the control
first BN standardized mean RMS distances are 4.99 (2100/2101), 6.30
(2100/2102), and 3.61 (2101/2102). The largest distance is between two
noncollapsed seeds. In the second BN, all control pairs have standardized mean
RMS distances around 2.23--2.40. Hybrid distances show the same pattern.

Thus seed 2101 is not identifiable by having the largest final raw BN distance.
These distances are descriptive because stem weights also differ by seed. The
buffer-only recalibration intervention supplies the causal evidence.

## Representation evidence

The final LayerNorm keeps mean target feature norms near 8.1 in every variant,
so collapse is not caused by exploding or vanishing feature magnitude. For
Hybrid seed 2101, the RMS of the target feature-channel mean changes from
0.941 under original mixed buffers to 0.304 under target recalibration. BN
therefore changes feature direction and centering seen by the fixed classifier,
which is consistent with a normalization/classifier alignment failure.

## Consequence for the next experiment

The priority should be a controlled BN-mixture stabilization study, not further
tokenizer decomposition. The current evidence motivates rules that prevent the
running state from drifting into the source-collapse regime while retaining
the useful majority calibration of mixed BN. Candidate interventions should be
predeclared and tested on all three seeds, such as domain-specific BN buffers,
fixed source/target moment blending, or freezing mixed buffers after an early
epoch. Target labels must not choose the blend or freeze time.

The frozen manifest SHA256 is
`649095aef54c3790791cc66f71effdce149ceb711f7ab2d9d78b22aba7152365`.
Artifacts are under `results/mixed_bn_stability/final_three_seed/`. Predictions,
features, BN snapshots, and hashes were frozen before target labels were read.
