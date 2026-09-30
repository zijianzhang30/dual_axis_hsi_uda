# Set-level coupled cross-domain retrieval

C0 and C1 use the locked A1 backbone: 3D stem → four spatial queries → class
query. Both arms instantiate the same additional interaction module so their
initialization and augmentation RNG remain comparable. C0 trains only A1
source CE. C1 uses one shared bidirectional cross-attention module:

- each target sample's four tokens query all `batch × 4` source tokens;
- each source sample's four tokens query all `batch × 4` target tokens;
- source self/cross CE terms are averaged;
- detached target cross predictions teach target self predictions through KL;
- inference uses only the unchanged A1 self branch.

There is no MMD, target consistency augmentation, diversity loss, spectral
query branch, pseudo-label loss, class/slot correspondence, or one-to-one
source-target binding. The cross residual uses one shared fixed scalar
`alpha=1`. A 20-epoch probe with learnable alpha initialized at 0.1 reduced
alpha to 0.080 and made the coupled branch approach the self branch, so the
formal control removes that bypass. Zero initialization is also avoided because
it blocks attention projection gradients on the first step.

The formal protocol is Houston13 → Houston18, patch 13, batch 128, SGD 0.01,
200 epochs, and seeds 2100/2101/2102. Selection is fixed epoch 200. The Strict
BiDA target GT mask determines eligible target centers; target label values do
not enter training or checkpoint selection.

Run all paired controls:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/set_coupled/run_formal.py \
  --device cuda:1
```
