# A2-lite: explicit spectral queries without dual-axis fusion

The experiment isolates whether an **explicit global spectral-token branch**
helps the source-only A1 representation. A1 already uses spectral information
through the 3D stem and spatial projection, so this is not an absent-vs-present
spectral-information test.

Both arms share the same stem and four spatial queries. A2-lite adds four
spectral queries over the 12 latent wavelength positions after spatial GAP.
The final class query reads the concatenated four spatial and four spectral
tokens. There is no spatial↔spectral fusion block, source-target interaction,
target loss, or pretrained weight. Existing A2 retains its two fusion blocks.

The paired arms are `a1` and `a2_lite`, each trained afresh for 200 epochs on
seeds 2100/2101/2102. Houston13 → Houston18 follows the Strict BiDA
normalization, 95/5 source split, 13×13 patch, batch 128, and SGD 0.01 with no
momentum/schedule. The target loader's GT mask chooses candidate centers,
but target labels do not enter the loss or checkpoint selection. Target OA/AA,
Kappa, and per-class recall are evaluated at **fixed epoch 200**.

The model exposes `class_attn_self`: four spatial weights followed by four
spectral weights. Their total spectral mass is diagnostic only; attention
weight alone is not a causal measure of useful information. Trainable
parameter counts are logged to show the capacity change from A1.

Run all six paired controls:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/a2_lite/run_formal.py \
  --device cuda:1
```
