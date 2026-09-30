# Dual-axis HSI UDA V0

Implements the [revised V0 protocol](../../docs/V0_REVISED_PROTOCOL.md). The Houston loader, `normband` normalization, 95/5 stratified source split, augmentation, 13×13 patches, and target sampling scheme come from `IEEE_TCSVT_BiDA/utils/dataset.py`. The primary checkpoint is fixed at epoch 200. The target training loader uses the target GT mask to select centers, as in Strict BiDA; target label values are discarded by the training loop. Target GT never enters a loss or checkpoint selection.

## Ablations

| Arm | Model |
| --- | --- |
| A0 | factorized 3D stem, global pool, classifier |
| A1 | stem, spatial semantic queries, class query |
| A2 | A1 plus spectral global queries and bidirectional within-sample fusion |
| A3 | A2 plus class × token-slot source EMA memory, source memory CE, and target memory-to-self distillation |

The latent cube is `[N, 64, 12, 13, 13]` by default. Attention uses four heads, pre-layer normalization, residuals, and GELU MLPs. GroupNorm is used in the stem. SGD uses `lr=0.01`, with no momentum or schedule, matching the Strict BiDA optimizer. Defaults use batch size 128, 200 epochs, and seeds 2100–2102.

**A3 training:** Each paired source/target step averages source self CE and source memory CE, then adds KL from detached target memory logits to target self logits. The KL weight and temperature are fixed at 1. The memory contains detached tokens from earlier source batches and updates after each optimizer step using source GT. The primary target inference and OA use the self branch only; the memory branch supplies training guidance and post-hoc diagnostics. `history.jsonl` records both CE components, their mean, target KL, both injection coefficients, and target self/memory prediction disagreement every epoch.

## Run

Use an environment with PyTorch, NumPy, SciPy, scikit-learn, `hdf5storage`, and `tifffile`, such as `/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python`.

```bash
cd /home/zhangzj26/dual_axis_hsi_uda
python experiments/v0/run_formal.py --device cuda:0
```

For a selected arm and seed, use `--variant a3 --seed 2100`. After all runs, aggregate:

```bash
python experiments/v0/summarize.py results/v0/a0_* results/v0/a1_* results/v0/a2_* results/v0/a3_*
```

Each run is saved under `results/v0/<arm>_<seed>/` with `final.pth`, `config.json`, `history.jsonl`, and `summary.json`. The controller executes this directory's `train.py`, `model.py`, and `data.py` snapshot, so later root-level edits do not change V0. `summary.json` contains OA, AA, Kappa, per-class recall, mean attention maps, A3 class attention, learned injection strengths, and post-hoc prediction changes. The full per-sample diagnostic tensors are exposed by `DualAxisModel.forward` in this directory's `model.py`.

Strict BiDA checkpoints under `IEEE_TCSVT_BiDA/experiments/bida_memory_v1/formal/strict_*/final.pth` can serve as historical controls, but their earlier Houston18 results were produced under a prior development protocol. Treat Houston18 comparisons as development evidence.
