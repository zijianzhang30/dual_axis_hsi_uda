# A1 spatial attention diversity gate

This is the first stage of the proposed slot specialization route. D0 is a
fresh A1 source-only control and D1 is the same A1 with a source attention-map
diversity penalty. Both use the same paired source/target loader iteration,
Houston13 → Houston18 normalization and splits, patch 13, batch 128, SGD 0.01,
200 epochs, and seeds 2100/2101/2102. Neither arm uses target images in a
loss. The released target GT mask still supplies eligible target centers to
the paired loader; target labels are used only for final evaluation.

For each sample, let `A` be the four spatial query attention maps over 169
positions. Normalize each row to unit L2 norm and calculate
`||A Aᵀ - I||²_F / [K(K−1)]`; this is the mean squared off-diagonal attention
cosine. D1 adds `0.1 × diversity` to source CE, with the coefficient fixed
before target evaluation. D0 computes the same diagnostic but multiplies it
by zero. The primary checkpoint is fixed epoch 200.

The orthogonal-attention penalty is an established regularizer (for example,
[Lin et al., 2017](https://arxiv.org/abs/1703.03130)); this study tests whether
it creates useful specialization in this HSI backbone.

The gate is evaluated with target OA/AA/Kappa/per-class recall and with both
attention-map cosine and token cosine on source validation and target test
patches. A lower attention overlap alone does not establish semantic roles:
the token outputs also need separation, and classification must remain stable.
The completed three-seed outcome and positional-attention audit are in
[`RESULTS.md`](RESULTS.md).

Run one short check:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/slot_diversity/train.py \
  --arm d1 --seed 2100 --epochs 1 --device cuda:1 \
  --out results/slot_diversity_smoke/d1_2100
```

Run the paired formal study:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/slot_diversity/run_formal.py \
  --device cuda:1
```
