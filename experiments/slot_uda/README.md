# Spatial slot UDA study

Houston13 → Houston18, Strict BiDA normalization/source split/target candidate mask,
patch 13, batch 128, SGD 0.01, 200 epochs, seeds 2100/2101/2102. The primary
checkpoint is always epoch 200; Houston18 labels enter evaluation only. The
released GT mask still determines eligible target training centers.

| Arm | Objective |
| --- | --- |
| B0 | A1 source CE; reuse `results/v0/a1_<seed>` |
| B1 | CE + global MMD on the mean of four spatial tokens |
| B2 | CE + the mean of four same-slot MMD terms |
| B3 | CE + target weak-to-strong prediction KL |
| B4 | CE + slot MMD + target prediction KL |

Both MMD arms use the same three RBF kernels, detached per-batch median bandwidth,
and biased MMD squared estimator. The slot loss averages four terms. Default
weights are fixed at `lambda_align=1`, `lambda_cons=1`; there is no target-label
tuning. The target weak view comes from the existing BiDA augmentation loader;
the strong view adds a center-preserving rotation and optional flip. Consistency
uses stopped-gradient weak predictions, temperature 1, and no EMA teacher.

The pre-run B0 audit found weak slot separation at epoch 200. On target patches,
mean off-diagonal token cosine is 0.980/0.971/0.940 and attention-map cosine is
0.969/0.976/0.934 for seeds 2100/2101/2102. Details are in
`results/slot_uda/b0_diagnostics_<seed>.json`. Accordingly, a B2 gain alone
cannot establish that the four slots represent distinct semantic factors.

Run one short check:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/slot_uda/train.py --arm b2 --seed 2100 --epochs 1 \
  --device cuda:1 --out results/slot_uda_smoke/b2_2100
```

Run the formal B1–B4 matrix:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/slot_uda/run_formal.py --device cuda:1
```

Each run writes `config.json`, `history.jsonl`, `final.pth`, and `summary.json`.
The summary includes OA, AA, Kappa, per-class recall, target prediction counts,
mean attention maps, and mean slot cosine matrices. Inspect B1/B2 first-epoch
raw MMD values to audit loss scale. Compare B2−B1 for alignment location and
B4−B2 together with B3−B0 for the consistency contribution.
