# Joint-BN tokenizer retrospective diagnosis

Identical label-unstratified fixed probes: 2048 patches per domain, batch 128.
Frozen checkpoint weights only. Eval uses saved buffers; joint-train probes
use current paired moments, dropout disabled, no gradients or updates.
Every state restoration was hash-checked. These are reconstructed probes,
not recorded historical training activations.

Direction concentration is the norm of the mean unit vector; higher means
more similar directions across samples. Cosine values are model-local.

## Fixed-200 eval target probes

| Seed | Tokenizer | BN2 variance | Token norm | Token/feature RMS ratio | Pre-LN CLS norm | z norm | z concentration | Cos(z,w6) | Class-6 margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | bida | 0.744 | 4.210 | 0.583 | 7.222 | 8.141 | 0.521 | 0.343 | 0.196 |
| 2100 | query | 0.719 | 2.588 | 0.361 | 6.787 | 8.120 | 0.554 | 0.333 | 0.293 |
| 2101 | bida | 0.719 | 4.403 | 0.614 | 7.083 | 8.151 | 0.618 | 0.447 | 2.184 |
| 2101 | query | 0.671 | 2.847 | 0.399 | 6.729 | 8.109 | 0.920 | 0.693 | 6.754 |
| 2102 | bida | 0.747 | 4.123 | 0.574 | 7.348 | 8.127 | 0.624 | 0.449 | 1.624 |
| 2102 | query | 0.706 | 2.779 | 0.391 | 6.589 | 8.104 | 0.715 | 0.435 | 0.541 |

## Seed 2101: eval trajectory

| Tokenizer | Epoch | BN2 variance | Token norm | Pre-LN CLS norm | z concentration | Cos(z,w6) | Class-6 margin | Probe class-6 share | Full-target class-6 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bida | 10 | 0.699 | 3.893 | 6.505 | 0.761 | 0.499 | 3.624 | 80.42% | 80.02% |
| bida | 20 | 0.774 | 4.188 | 6.370 | 0.527 | 0.305 | 0.416 | 55.66% | 54.70% |
| bida | 30 | 0.709 | 4.020 | 6.637 | 0.687 | 0.463 | 2.774 | 68.90% | 67.68% |
| bida | 40 | 0.729 | 4.364 | 6.730 | 0.634 | 0.446 | 2.255 | 65.38% | 64.00% |
| bida | 50 | 0.758 | 4.193 | 6.808 | 0.651 | 0.410 | 2.083 | 63.62% | 62.54% |
| bida | 60 | 0.744 | 4.352 | 6.907 | 0.656 | 0.428 | 2.200 | 63.96% | 62.75% |
| bida | 70 | 0.765 | 4.271 | 6.813 | 0.583 | 0.361 | 1.070 | 57.03% | 56.80% |
| bida | 80 | 0.754 | 4.308 | 6.893 | 0.601 | 0.390 | 1.398 | 58.40% | 57.90% |
| bida | 90 | 0.734 | 4.236 | 6.900 | 0.647 | 0.440 | 2.178 | 63.09% | 62.07% |
| bida | 100 | 0.718 | 4.187 | 6.933 | 0.664 | 0.453 | 2.351 | 63.77% | 62.52% |
| bida | 110 | 0.749 | 4.361 | 7.092 | 0.677 | 0.469 | 2.620 | 64.60% | 63.53% |
| bida | 120 | 0.757 | 4.336 | 7.000 | 0.609 | 0.405 | 1.473 | 57.62% | 57.05% |
| bida | 130 | 0.762 | 4.412 | 7.013 | 0.610 | 0.400 | 1.452 | 57.62% | 57.09% |
| bida | 140 | 0.777 | 4.463 | 7.011 | 0.575 | 0.367 | 0.918 | 55.66% | 54.84% |
| bida | 150 | 0.735 | 4.413 | 7.006 | 0.619 | 0.434 | 2.082 | 60.84% | 60.15% |
| bida | 160 | 0.744 | 4.462 | 7.042 | 0.612 | 0.418 | 1.789 | 59.42% | 58.61% |
| bida | 170 | 0.749 | 4.545 | 7.072 | 0.615 | 0.412 | 1.717 | 58.89% | 58.25% |
| bida | 180 | 0.760 | 4.546 | 7.028 | 0.585 | 0.393 | 1.370 | 57.47% | 56.75% |
| bida | 190 | 0.758 | 4.550 | 7.015 | 0.555 | 0.356 | 0.759 | 54.79% | 53.98% |
| bida | 200 | 0.719 | 4.403 | 7.083 | 0.618 | 0.447 | 2.184 | 62.21% | 61.46% |
| query | 10 | 0.616 | 2.589 | 6.264 | 0.859 | 0.454 | 3.953 | 90.92% | 90.99% |
| query | 20 | 0.956 | 2.801 | 5.393 | 0.576 | -0.054 | -4.240 | 23.97% | 23.52% |
| query | 30 | 0.640 | 2.748 | 6.479 | 0.899 | 0.566 | 5.176 | 92.14% | 92.65% |
| query | 40 | 0.632 | 2.866 | 6.451 | 0.874 | 0.553 | 4.916 | 88.09% | 88.27% |
| query | 50 | 0.691 | 2.867 | 6.607 | 0.899 | 0.585 | 5.445 | 90.48% | 90.96% |
| query | 60 | 0.658 | 2.900 | 6.652 | 0.919 | 0.634 | 6.041 | 96.39% | 96.50% |
| query | 70 | 0.668 | 2.870 | 6.653 | 0.916 | 0.644 | 6.115 | 96.00% | 95.78% |
| query | 80 | 0.677 | 2.875 | 6.659 | 0.915 | 0.648 | 6.132 | 95.26% | 95.24% |
| query | 90 | 0.679 | 2.881 | 6.685 | 0.917 | 0.659 | 6.293 | 95.90% | 95.80% |
| query | 100 | 0.671 | 2.863 | 6.671 | 0.915 | 0.664 | 6.317 | 95.51% | 95.38% |
| query | 110 | 0.667 | 2.887 | 6.690 | 0.921 | 0.674 | 6.503 | 96.24% | 96.40% |
| query | 120 | 0.666 | 2.935 | 6.738 | 0.917 | 0.677 | 6.428 | 94.58% | 94.67% |
| query | 130 | 0.661 | 2.909 | 6.753 | 0.919 | 0.675 | 6.408 | 94.92% | 95.10% |
| query | 140 | 0.679 | 2.891 | 6.716 | 0.916 | 0.671 | 6.413 | 94.24% | 94.66% |
| query | 150 | 0.673 | 2.909 | 6.747 | 0.921 | 0.684 | 6.627 | 95.56% | 95.69% |
| query | 160 | 0.626 | 2.886 | 6.557 | 0.876 | 0.651 | 5.766 | 92.43% | 92.55% |
| query | 170 | 0.660 | 2.864 | 6.644 | 0.915 | 0.677 | 6.508 | 97.71% | 98.05% |
| query | 180 | 0.665 | 2.851 | 6.687 | 0.917 | 0.683 | 6.552 | 97.66% | 98.05% |
| query | 190 | 0.661 | 2.864 | 6.709 | 0.923 | 0.692 | 6.737 | 97.71% | 98.09% |
| query | 200 | 0.671 | 2.847 | 6.729 | 0.920 | 0.693 | 6.754 | 97.56% | 97.90% |

## Seed 2101: fixed-200 source/target and mode comparison

| Tokenizer | Mode | Domain | BN2 abs mean | BN2 variance | Token norm | z concentration | Cos(z,w6) | Class-6 margin | Class-6 share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bida | eval | source | 0.230 | 1.114 | 4.290 | 0.142 | 0.080 | -6.895 | 15.92% |
| bida | eval | target | 0.255 | 0.719 | 4.403 | 0.618 | 0.447 | 2.184 | 62.21% |
| bida | joint_train | source | 0.245 | 1.117 | 4.312 | 0.134 | 0.061 | -7.119 | 15.92% |
| bida | joint_train | target | 0.244 | 0.728 | 4.405 | 0.600 | 0.430 | 1.888 | 60.50% |
| query | eval | source | 0.298 | 1.117 | 2.480 | 0.143 | 0.057 | -6.997 | 15.92% |
| query | eval | target | 0.306 | 0.671 | 2.847 | 0.920 | 0.693 | 6.754 | 97.56% |
| query | joint_train | source | 0.300 | 1.102 | 2.471 | 0.143 | 0.056 | -6.993 | 15.92% |
| query | joint_train | target | 0.301 | 0.666 | 2.849 | 0.922 | 0.695 | 6.789 | 98.10% |

## Evidence-backed interpretation

Seed 2101 does not show token-norm explosion: at epoch 200 Query token norm is 2.847 versus BiDA 4.403. BN2 target variance is 0.671 versus 0.719; classifier-input z norm is 8.109 versus 8.151.
The stronger signature is target direction concentration after the downstream transformer. Token direction concentration is relatively close (Query 0.868, BiDA 0.851), whereas pre-final-LayerNorm CLS concentration is 0.920 versus 0.618. The final LayerNorm preserves this difference; it does not create it at that boundary.
Target cosine to the local class-6 classifier weight rises to 0.693 versus 0.447. Class-6 weight norm/bias do not explode: Query 1.333/0.00908 versus BiDA 1.368/0.00952. Mean target class-6 logits are 7.50 versus 5.00; margins over the best other logit are 6.75 versus 2.18.
The query classifier representation remains diverse on source (concentration 0.143) but concentrates on target (0.920). This is domain-specific representational collapse, not a globally broken classifier.
Query target directional concentration is already elevated at epoch 10 (0.859), reverses at epoch 20, then remains high from epoch 30 (0.899) with margin 5.18. Full-target class-6 share first crosses the predeclared 95% collapse threshold at epoch 60. Therefore there is no single monotonically increasing early onset; the sustained separation starts around the sampled epoch-30 checkpoint.
Current joint-training moments do not restore diversity at epoch 200: Query target concentration 0.922, margin 6.79, class-6 share 98.10%; eval share is 97.56% on the same sample. This is not solely saved running-buffer mismatch.
The observations localize the pronounced concentration change somewhere between tokenizer outputs and pre-final-LayerNorm CLS features in the learned tokenizer-plus-transformer system. They do not prove that the query reader alone rotates features or that a residual/normalization change will repair it. Per-block or controlled-intervention evidence would be needed for that claim.
Retain Full Joint + BiDA tokenizer as the reference. No rescue tuning or query normalization changes were made.

## Limitations

BN statistics are per-channel averages of pre-ReLU outputs. Token norms are
after different tokenizer packages; their absolute scales are not intrinsically
comparable as a defect. The token/feature ratio is descriptive, not a causal gain.
Train-mode probes disable dropout and omit historical augmentation; the sample
is not class-stratified. We do not use GT labels for grouping or sampling.
Observed scale/direction changes do not identify the tokenizer as the sole cause:
the stem, normalization affine parameters, transformer and classifier co-adapt.
No architecture changes or normalization fixes were made. Full JSON includes
per-channel vectors, classifier norms/biases, all source/target modes and seeds.
