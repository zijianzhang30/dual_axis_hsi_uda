# Query PE-off replication: seeds 2100 / 2101 / 2102

The two new replications completed; seed 2101 is reused unchanged from the
previous intervention. Seed 2100 was already running on GPU 6 when GPU 4/5
were requested, so it was allowed to finish; seed 2102 ran on GPU 4. Future
jobs should prefer GPU 4/5 when free. No duplicate run or Key-only PE was started.

All results use the fixed epoch-200 endpoint. Means +/- SD use sample SD
(ddof=1). Save grid remains 10:10:200, with all predictions frozen before
post-hoc target metrics. Initial tensors and shared component hashes match
each paired PE-on control. The PE-off model is depth 0 + Class Query, not
the depth-3 constant-CLS Hybrid-V2 model.

## Fixed-200 per-seed results

| Seed | PE-off OA | PE-off AA | Kappa | Class-6 share | Zero-recall classes | OA off minus on | AA off minus on |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 2100 | 77.84% | 65.50% | 0.637 | 57.73% | None | -2.63 pp | -5.18 pp |
| 2101 | 79.02% | 65.46% | 0.635 | 64.83% | None | +4.47 pp | +18.22 pp |
| 2102 | 81.77% | 70.79% | 0.685 | 63.97% | None | +0.35 pp | +1.61 pp |

| Model | OA mean +/- SD | AA mean +/- SD | Kappa mean +/- SD | Class-6 collapse |
| --- | ---: | ---: | ---: | ---: |
| Query + Class Query, PE-on, depth 0 | 78.81 +/- 3.72% | 62.37 +/- 13.12% | 0.604 +/- 0.119 | 0/3 |
| Query + Class Query, PE-off, depth 0 | 79.54 +/- 2.02% | 67.25 +/- 3.06% | 0.652 +/- 0.028 | 0/3 |
| Full Joint + BiDA tokenizer, depth 3 reference | 79.50 +/- 1.46% | 71.90 +/- 1.72% | 0.665 +/- 0.024 | 0/3 |

Collapse uses the existing >=95% class-6 prediction-share definition. PE-on
already has 0/3 under this threshold, despite seed 2101 losing class 7 entirely;
do not claim the thresholded collapse count improved. PE-off restores nonzero
fixed-200 recall for all seven classes in all three seeds and reduces observed
OA/AA variation. Minority-class recall can still vary substantially across seeds.

The paired average effect is +0.73 pp OA and +4.88 pp AA, with sample SD of
paired effects 3.57 and 12.04 pp respectively. Effects are heterogeneous;
removing PE is not uniformly beneficial (seed 2100 becomes worse).

## Per-class recall, fixed 200

| Class | 2100 | 2101 | 2102 | Mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29.79% | 81.15% | 57.65% | 56.20 +/- 25.71% |
| 2 | 55.54% | 78.58% | 76.61% | 70.24 +/- 12.77% |
| 3 | 67.87% | 62.04% | 65.70% | 65.21 +/- 2.95% |
| 4 | 81.82% | 54.55% | 81.82% | 72.73 +/- 15.75% |
| 5 | 73.74% | 65.70% | 78.36% | 72.60 +/- 6.41% |
| 6 | 87.86% | 93.71% | 93.23% | 91.60 +/- 3.25% |
| 7 | 61.90% | 22.51% | 42.16% | 42.19 +/- 19.70% |

## Prediction distribution, fixed 200

Each seed predicts 52,901 samples. These are predicted classes, not true class
frequencies; the full JSON contains exact counts, shares and confusion matrices.

| Predicted class | 2100 count / share | 2101 count / share | 2102 count / share |
| --- | ---: | ---: | ---: |
| 1 | 600 / 1.13% | 2294 / 4.34% | 1125 / 2.13% |
| 2 | 2916 / 5.51% | 6604 / 12.48% | 4309 / 8.15% |
| 3 | 4020 / 7.60% | 2501 / 4.73% | 3603 / 6.81% |
| 4 | 1028 / 1.94% | 12 / 0.02% | 31 / 0.06% |
| 5 | 7015 / 13.26% | 5354 / 10.12% | 6289 / 11.89% |
| 6 | 30542 / 57.73% | 34295 / 64.83% | 33843 / 63.97% |
| 7 | 6780 / 12.82% | 1841 / 3.48% | 3701 / 7.00% |

## Oracle diagnostic and checkpoint coverage

| Seed | Oracle OA | Corresponding epoch | Minimum grid OA |
| --- | ---: | ---: | ---: |
| 2100 | 79.52% | 20 | 77.61% |
| 2101 | 80.20% | 60 | 78.00% |
| 2102 | 81.97% | 140 | 78.69% |

Oracle OA is 80.56 +/- 1.26%, diagnostic only. None of the 60 saved PE-off
checkpoints crosses the class-6-collapse threshold. Seed 2101 epoch 50 has a
zero-recall class, so full class coverage is not guaranteed at every checkpoint.
Do not choose the oracle checkpoint or hyperparameters based on these outcomes.

## Decision and mechanism boundary

PE-off reaches the reference's OA level within 0.04 pp, but its OA SD is higher
and AA is 4.65 pp lower. It is a useful improved-coverage Query candidate,
not evidence that it should replace the Full Joint + BiDA tokenizer reference.
The reference differs in depth/readout, so this comparison is not a strict
single-component tokenizer contrast.

The stress-seed rescue replicated as lower observed across-seed variation and
full fixed-endpoint coverage, not as a positive accuracy effect in every seed.
Three seeds, including an already examined stress seed, are limited evidence.
Do not claim positional encodings are universally harmful or that PE-in-Value
is the established mechanism: PE-off changes both keys and values, and all
weights co-adapt during training. The previously proposed Key-only contrast
K=LN(X+PE), V=LN(X) remains a suitable direct hypothesis test. It was not run
in this extension; no LN/attention detail sweep or new UDA module was added.

## Artifacts and checks

- Aggregate: `results/query_pe_off_multiseed/summary.json`
- New frozen runs: `results/query_pe_off_multiseed/formal_2100/` and `formal_2102/`
- Reused frozen run: `results/query_pe_off_v1/formal_2101/`
- Strict paired controls: `results/query_class_readout_v1/formal_{seed}/`

All 240 paired PE-on/off checkpoint and prediction file hashes, all six
manifest hashes, initialization/config equality checks and 20-checkpoint grids
passed. The existing 2101 terminal-print correction/provenance is documented
in `experiments/query_pe_off_v1/RESULTS.md`; no historical artifact was rewritten.
