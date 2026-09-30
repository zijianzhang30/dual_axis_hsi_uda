# A0–A2 maximum Houston18 OA under the released BiDA selection rule

The released BiDA trainer evaluates target test OA every 10 epochs and keeps the highest checkpoint. We applied that rule to **independent reruns** of V0 A0, A1, and A2 for seeds 2100–2102. Each run trained 200 epochs, saved epochs 10, 20, ..., 200, and evaluated the same 52,901 Houston18 pixels after training. Ties use the earliest epoch. Target labels select the reported checkpoint, so these values are **target-oracle diagnostics**, not a valid target-blind checkpoint selection or a replacement for the [fixed-epoch V0 result](../../experiments/v0/RESULTS.md).

| Arm | Seed | Best epoch | Highest target OA | Same rerun, epoch-200 OA | Gain from oracle selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 2100 | 150 | 65.28% | 59.13% | +6.15 points |
| A0 | 2101 | 150 | 60.91% | 57.90% | +3.01 points |
| A0 | 2102 | 120 | 64.35% | 55.93% | +8.42 points |
| **A0 mean ± SD** | | | **63.52 ± 2.30%** | **57.65 ± 1.61%** | **+5.86 points** |
| A1 | 2100 | 60 | 74.16% | 71.92% | +2.24 points |
| A1 | 2101 | 200 | 71.58% | 71.58% | +0.00 points |
| A1 | 2102 | 30 | 72.65% | 69.71% | +2.95 points |
| **A1 mean ± SD** | | | **72.80 ± 1.30%** | **71.07 ± 1.19%** | **+1.73 points** |
| A2 | 2100 | 30 | 72.57% | 68.65% | +3.92 points |
| A2 | 2101 | 200 | 71.97% | 71.97% | +0.00 points |
| A2 | 2102 | 40 | 76.03% | 71.10% | +4.93 points |
| **A2 mean ± SD** | | | **73.52 ± 2.19%** | **70.57 ± 1.72%** | **+2.95 points** |

Under this oracle rule, paired OA differences are `A1−A0 = +8.88, +10.67, +8.30` points and `A2−A1 = −1.59, +0.39, +3.38` points for seeds 2100–2102. Mean A2−A1 is **+0.72 points**; it is mixed across seeds. The best single A2 result is **76.03%**, below the BiDA paper's Houston18 **81.11%**. The local released-code BiDA reproduction reached **82.1553%** in one seed at epoch 60 under the same target-OA oracle rule; this is a contextual result for a different model and training objective.

## Replay and interpretation limits

Every rerun has 200 training records, 20 saved checkpoints, and 20 target evaluation records. Its paired batch count matches the original. Exact trajectory replay failed: a same-seed attempt drifted in source training loss by epoch 4–5 even on the original GPU. The nine completed runs therefore use **new training trajectories**. Compare each oracle maximum only with the **same rerun's** epoch-200 OA; do not subtract it from the original V0 final-OA table as if they were checkpoints of one run. The per-epoch target metrics, checkpoints, confusion matrices, and replay-difference logs are under [`results/v0_target_oracle/`](../../results/v0_target_oracle/).

The source 95/5 split remains fixed at the BiDA loader's random state 23. The Strict BiDA target loader uses the Houston18 GT mask to define eligible training centers. Target class values do not enter the source-only A0–A2 losses, but target GT is used here to select the best epoch. Houston18 has already served as a development benchmark; this oracle maximum is an upper-bound diagnostic and should not guide a claimed target-blind method improvement.
