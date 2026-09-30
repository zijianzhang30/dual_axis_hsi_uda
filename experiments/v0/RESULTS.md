# V0 Houston13 → Houston18 results

All A0–A3 runs completed 200 epochs for seeds 2100, 2101, and 2102. Primary evaluation uses the fixed epoch-200 **target self** branch on 52,901 Houston18 labeled pixels. No target metric selected a checkpoint. The [run summaries](../../results/v0/) and [aggregate metrics](../../results/v0/aggregate.json) contain OA, AA, Kappa, per-class recall, confusion matrices, and attention diagnostics.

## Primary result

Target OA (%):

| Seed | A0 stem | A1 spatial | A2 dual-axis | A3 memory + distillation | A2 − A1 | A3 − A2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 65.06 | 71.52 | 64.88 | 68.95 | −6.63 | +4.06 |
| 2101 | 53.91 | 71.35 | 71.35 | 6.03 | −0.01 | −65.32 |
| 2102 | 59.96 | 68.28 | 69.37 | 16.91 | +1.09 | −52.45 |
| Mean ± sample SD | 59.64 ± 5.58 | 70.38 ± 1.82 | 68.53 ± 3.31 | 30.63 ± 33.63 | −1.85 ± 4.18 | −37.90 ± 36.91 |

Mean AA (%) is 38.12, 55.56, 58.75, and 37.93 for A0–A3 respectively. Mean Kappa is 0.315, 0.404, 0.437, and 0.160. Spatial tokenization improves OA over the stem in all three seeds. Adding spectral global tokens does not improve mean OA, although mean AA rises by 3.20 points. The full A3 does not provide stable cross-scene benefit: it beats A2 only in seed 2100 and collapses in the other two.

## A3 mechanism diagnostics

| Seed | Final α spatial | Final α spectral | Final train target KL | Final train prediction disagreement | Full-target memory/self disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2100 | 0.00837 | 0.00979 | 4.36×10⁻⁷ | 0.000% | 0.0095% |
| 2101 | 0.00723 | 0.00991 | 4.51×10⁻⁷ | 0.000% | 0.0246% |
| 2102 | 0.00935 | 0.00736 | 6.08×10⁻⁷ | 0.000% | 0.0113% |

The maximum epoch-mean target KL over all 200 epochs is below `4.5×10⁻⁶` in every seed. The full-target memory branch changes only 5, 13, and 6 predictions, respectively; it corrects none of the self branch errors and makes 3, 10, and 3 correct self predictions wrong. Thus the memory-guidance signal is extremely weak under the near-zero injection initialization. This observation does **not** prove that the tiny KL caused A3's target collapse. The source self representation itself transfers poorly in seeds 2101 and 2102: their target self predictions put 96.4% and 79.9% of pixels into classes 0/3 and class 0, respectively, despite perfect source validation OA at epoch 200.

## Context and limits

Historical Strict BiDA fixed-epoch-200 OA for these optimization seeds is 78.75%, 64.51%, and 77.24% (mean 73.50%). These checkpoints are contextual, not a new paired control: Strict BiDA uses a different network and training objective. Houston18 was already a development benchmark before V0.

The source 95/5 split is fixed by the BiDA loader's random state 23; the three runs vary optimization seed, not source split. Strict BiDA's target training loader uses the Houston18 GT **mask** to choose eligible unlabeled centers. Target class values do not enter V0 losses or checkpoint selection, but target GT does affect training sample selection. A0–A2 consume paired target batches without a target loss; A3 uses those images in memory-to-self KL. The source CE average was fixed before formal runs so identical self and memory logits yield the same source gradient scale as A2.

**V0 conclusion:** This implementation does not establish either proposed gain. Spectral global tokens show mixed OA effects; the A3 memory/distillation mechanism supplies almost no observable teacher signal and yields unstable target self transfer. Diagnose this V0 behavior before proposing new objectives or target-GT-tuned changes.

A separate [released-BiDA-style target-OA oracle investigation](../../investigations/target_oracle_v0/RESULTS.md) reruns A0–A2 and reports their maximum Houston18 OA among epochs 10–200 in steps of 10. Those target-selected maxima are not the fixed-epoch primary result.
