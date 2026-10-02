# Noisy SceneShift x Full Joint BN: fixed-200 2x2 results

Twelve fresh runs completed: Houston13 -> Houston18, original BiDA spatial-softmax
tokenizer, depth 3, dropout .1, SGD .01, 200 epochs, seeds 2100/2101/2102.
Original BN means the native sequential source -> target mixed-BN recipe, not
source-only BN. Full Joint retains the differentiable joint recipe with momentum .19.

SceneShift is the frozen historical alpha .7 noisy recipe: epsilon 1e-5, amplitude
Gaussian SD .04, smooth Gaussian SD .015 via 5x5 average pooling, clamp [0,1],
extra .5 source CE. B/D loss is CE_s + .5 CE_shift; auxiliary BN updates normally.
Statistics use all unpadded normalized scene pixels, including background, without a GT mask.
The current input is BiDA normband, not historical MLUDA ILDA; the backbone/input-space
adaptation is disclosed and old MLUDA gains are not reused as evidence for this transfer.

## Outcome

No-Go for unifying the two factors under the preregistered practical gate.
SceneShift alone B-A has OA gains in 2/3 seeds and AA gains in 3/3.
Full Joint alone C-A has OA gains in 3/3 seeds and AA gains in 3/3.
The key D-C comparison has mean OA delta -1.91 pp and AA delta -0.39 pp.
The original stress-seed collapse contributes strongly to large B-A/C-A mean gains;
do not confuse recovery of a failed baseline seed with uniform OA gains in every seed.
For these seeds the composite noisy augmentation does not stably improve the joint-BN
self model. This is evidence against practical complementarity in the tested setting,
not a unique proof that normalization has absorbed every possible shift mechanism.

Shift noise and auxiliary dropout use separate persistent RNG streams, preserving
the original-forward RNG. Fixed epoch 200 is the sole target endpoint. No target-oracle
selection, target-selected checkpoint, alpha tuning, new loss/module or pure-affine arm.

## Primary three-seed aggregate

Sample SD uses ddof=1. Class-6 collapse is >=95% predicted class-6 share.

| Arm | OA (%) | AA (%) | Kappa | Class-6 share (%) | Collapse | Source-val OA (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: Original BN | 73.18 +/- 8.21 | 54.05 +/- 18.46 | 0.440 +/- 0.280 | 77.49 +/- 16.62 | 1/3 | 73.75 +/- 5.91 |
| B: Original + noisy shift | 78.53 +/- 1.63 | 72.77 +/- 2.45 | 0.653 +/- 0.023 | 56.16 +/- 2.62 | 0/3 | 71.92 +/- 4.04 |
| C: Full Joint | 79.50 +/- 1.46 | 71.90 +/- 1.72 | 0.665 +/- 0.024 | 57.21 +/- 4.79 | 0/3 | 100.00 +/- 0.00 |
| D: Full Joint + noisy shift | 77.59 +/- 0.81 | 71.51 +/- 0.55 | 0.648 +/- 0.011 | 52.30 +/- 0.49 | 0/3 | 100.00 +/- 0.00 |

## All fixed-endpoint seeds

| Arm | Seed | OA | AA | Kappa | C6 share | Source-val OA | Zero-recall classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 2100 | 78.41 | 66.32 | 0.627 | 63.66% | 79.53 | None |
| A | 2101 | 63.72 | 32.82 | 0.119 | 95.93% | 74.02 | 2, 7 |
| A | 2102 | 77.40 | 63.01 | 0.575 | 72.88% | 67.72 | None |
| B | 2100 | 77.26 | 75.16 | 0.642 | 53.16% | 68.50 | None |
| B | 2101 | 77.96 | 70.26 | 0.638 | 57.97% | 70.87 | None |
| B | 2102 | 80.36 | 72.88 | 0.679 | 57.36% | 76.38 | None |
| C | 2100 | 78.55 | 70.03 | 0.663 | 52.02% | 100.00 | None |
| C | 2101 | 78.77 | 72.25 | 0.641 | 61.46% | 100.00 | None |
| C | 2102 | 81.19 | 73.42 | 0.690 | 58.16% | 100.00 | None |
| D | 2100 | 77.00 | 71.50 | 0.641 | 51.79% | 100.00 | None |
| D | 2101 | 77.25 | 70.96 | 0.643 | 52.34% | 100.00 | None |
| D | 2102 | 78.52 | 72.07 | 0.661 | 52.77% | 100.00 | None |

Zero-recall classes are reported separately: avoiding the class-6 threshold is not
equivalent to retaining every class. Source-val OA is diagnostic only, not model selection.

## Paired effects (left minus right)

| Contrast | Seed | OA delta (pp) | AA delta (pp) | Kappa delta |
| --- | ---: | ---: | ---: | ---: |
| B-A | 2100 | -1.15 | +8.83 | +0.015 |
| B-A | 2101 | +14.24 | +37.44 | +0.519 |
| B-A | 2102 | +2.96 | +9.87 | +0.104 |
| B-A | mean +/- SD | 5.35 +/- 7.97 | 18.72 +/- 16.23 | 0.213 +/- 0.269 |
| C-A | 2100 | +0.14 | +3.71 | +0.036 |
| C-A | 2101 | +15.05 | +39.43 | +0.522 |
| C-A | 2102 | +3.78 | +10.41 | +0.115 |
| C-A | mean +/- SD | 6.33 +/- 7.77 | 17.85 +/- 18.99 | 0.224 +/- 0.261 |
| D-C | 2100 | -1.55 | +1.47 | -0.022 |
| D-C | 2101 | -1.52 | -1.29 | +0.001 |
| D-C | 2102 | -2.67 | -1.35 | -0.029 |
| D-C | mean +/- SD | -1.91 +/- 0.65 | -0.39 +/- 1.61 | -0.016 +/- 0.016 |

Positive-seed counts:
- B-A: OA 2/3; AA 3/3.
- C-A: OA 3/3; AA 3/3.
- D-C: OA 0/3; AA 1/3.

The factorial interaction (D-C)-(B-A) is:
- OA pp: -7.26 +/- 7.81.
- AA pp: -19.11 +/- 17.11.
- Kappa: -0.229 +/- 0.254.

Interaction is descriptive for these seeds. A positive interaction does not by itself
establish that D improves C, and a negative/zero D-C does not uniquely prove BN
has covered all benefits of input shift: the frozen SceneShift factor also includes
extra supervised CE, noise/clamp and an extra running-buffer update.

## Per-class recall (%)

### A: Original BN

| Class | 2100 | 2101 | 2102 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40.58 | 4.51 | 56.61 | 33.90 +/- 26.69 |
| 2 | 77.24 | 0.00 | 74.76 | 50.67 +/- 43.90 |
| 3 | 57.81 | 29.05 | 50.27 | 45.71 +/- 14.91 |
| 4 | 81.82 | 81.82 | 81.82 | 81.82 +/- 0.00 |
| 5 | 93.73 | 15.09 | 81.59 | 63.47 +/- 42.34 |
| 6 | 90.26 | 99.25 | 95.28 | 94.93 +/- 4.51 |
| 7 | 22.84 | 0.00 | 0.73 | 7.86 +/- 12.98 |

### B: Original + noisy shift

| Class | 2100 | 2101 | 2102 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 80.64 | 53.95 | 68.81 | 67.80 +/- 13.37 |
| 2 | 52.81 | 55.86 | 61.87 | 56.85 +/- 4.61 |
| 3 | 64.87 | 58.42 | 67.40 | 63.56 +/- 4.63 |
| 4 | 100.00 | 100.00 | 81.82 | 93.94 +/- 10.50 |
| 5 | 93.07 | 95.36 | 89.99 | 92.80 +/- 2.70 |
| 6 | 84.53 | 88.56 | 88.80 | 87.30 +/- 2.40 |
| 7 | 50.19 | 39.67 | 51.48 | 47.11 +/- 6.48 |

### C: Full Joint

| Class | 2100 | 2101 | 2102 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 30.75 | 74.94 | 71.18 | 58.96 +/- 24.50 |
| 2 | 73.66 | 76.06 | 79.77 | 76.49 +/- 3.08 |
| 3 | 60.12 | 63.78 | 50.85 | 58.25 +/- 6.67 |
| 4 | 81.82 | 86.36 | 81.82 | 83.33 +/- 2.62 |
| 5 | 93.65 | 92.88 | 96.81 | 94.45 +/- 2.08 |
| 6 | 82.49 | 89.29 | 89.01 | 86.93 +/- 3.85 |
| 7 | 67.75 | 22.43 | 44.49 | 44.89 +/- 22.66 |

### D: Full Joint + noisy shift

| Class | 2100 | 2101 | 2102 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 61.79 | 52.92 | 54.40 | 56.37 +/- 4.75 |
| 2 | 51.05 | 62.22 | 60.00 | 57.76 +/- 5.92 |
| 3 | 68.38 | 66.10 | 65.99 | 66.82 +/- 1.35 |
| 4 | 81.82 | 81.82 | 81.82 | 81.82 +/- 0.00 |
| 5 | 91.08 | 89.09 | 90.74 | 90.30 +/- 1.07 |
| 6 | 82.54 | 82.51 | 83.42 | 82.83 +/- 0.52 |
| 7 | 63.87 | 62.06 | 68.10 | 64.68 +/- 3.10 |

Exact predicted counts/shares and confusion matrices for every seed are preserved
in each results.json and aggregate summary.json.

## Parameters and cost

All arms have 376,567 trainable parameters; SceneShift and joint BN add zero parameters.
All arms use one optimizer update per paired batch. B/D add one supervised paired
forward and its backward gradient contribution; A/C have no auxiliary forward.

| Arm | Preparation seconds | Training seconds | Auxiliary CUDA span seconds |
| --- | ---: | ---: | ---: |
| A | 0.96 +/- 1.18 | 203.36 +/- 3.56 | 0.00 +/- 0.00 |
| B | 0.40 +/- 0.31 | 291.98 +/- 2.25 | 27.04 +/- 0.14 |
| C | 0.13 +/- 0.01 | 212.37 +/- 1.53 | 0.00 +/- 0.00 |
| D | 0.44 +/- 0.09 | 366.76 +/- 1.89 | 25.10 +/- 0.12 |

Measured paired additional training time:
- B-A: 88.62 +/- 3.81 seconds; left/right ratio 1.44 +/- 0.03.
- C-A: 9.01 +/- 5.09 seconds; left/right ratio 1.04 +/- 0.03.
- D-C: 154.39 +/- 3.42 seconds; left/right ratio 1.73 +/- 0.02.

Twelve runs were parallelized with three processes per free GPU: A on 4, B on 5,
C on 6, D on 7. Wall times include hashing/logging, validation and checkpoints.
CUDA event spans include noise generation + auxiliary forward/loss, not auxiliary
backward time in isolation. Concurrent kernel interleaving and different finishing
times mean these are observed workload costs, not controlled standalone speed benchmarks.

## Preregistered Go/No-Go

Before outcomes, material improvement was operationalized as mean D-C >=1 pp in
OA or AA and positive changes in at least 2/3 seeds in that same metric, with
no increase in collapse count. This is an exploratory practical gate, not significance.

Decision: **No-Go**.
- OA material/majority gate: False.
- AA material/majority gate: False.
- No collapse count increase: True.

Continue no automatic new module/variant from this run. Do not continue stacking SceneShift or launch alpha/pure-affine rescue experiments; retain Full Joint reference.

## Audit and frozen artifacts

All 252 checkpoint/prediction hashes and twelve manifest hashes passed.
Every seed has identical initial tensors, source/target indices and full-scene
statistics across all four arms. All 200 epochs have identical paired input stream
and pre-original-forward CPU/CUDA RNG stream hashes across A/B/C/D.

| Control | Seed | Historical final state exact | Maximum tensor difference |
| --- | ---: | --- | ---: |
| A | 2100 | True | 0 |
| A | 2101 | True | 0 |
| A | 2102 | True | 0 |
| C | 2100 | True | 0 |
| C | 2101 | True | 0 |
| C | 2102 | True | 0 |

Fresh A/C endpoints, not historical metrics, are used in formal paired contrasts.
Target GT-mask sampling is the unchanged BiDA loader behavior; label values do not
enter loss/checkpoint selection. Target prediction arrays and hashes were frozen
before post-hoc target GT scoring. No target labels were used to tune the noisy recipe.

Artifacts: results/sceneshift_joint_v1/{A,B,C,D}_{2100,2101,2102}/,
results/sceneshift_joint_v1/summary.json. Historical runs/files were preserved.
