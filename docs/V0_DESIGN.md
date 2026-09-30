# Cross-Scene HSI UDA — Dual-Axis Semantic Memory Network (V0)

## 1. Goal

Build a new HSI cross-scene UDA model under a **BiDA-style data/training pipeline**, while replacing the BiDA model itself.

The V0 model should test one main idea:

> **Keep the spectral axis explicit, learn spatial and spectral semantic tokens separately, then perform class-aware source-to-target interaction in the semantic token space.**

Do **not** add Flow Matching, OT, pseudo-label losses, contrastive losses, MMD/LMMD, or external priors in V0.

The purpose of V0 is to determine whether the proposed representation and interaction mechanism itself is useful.

---

## 2. Experimental protocol

Use the same strict BiDA-style pipeline already used for our BiDA reproduction.

Initial benchmark:
- Houston13 → Houston18
- patch size: `13 × 13`
- same normalization as Strict BiDA
- same source/target split and loaders as Strict BiDA
- same batch size, optimizer, learning-rate schedule, seeds, and 200-epoch training budget unless a change is strictly required by implementation
- **Primary checkpoint: fixed epoch 200**
- target GT must not be used for training or checkpoint selection

Initial comparison:
1. Strict BiDA
2. our encoder-only baseline
3. proposed V0

Later validation:
- MLUDA benchmark protocol
- DAMamba benchmark protocol
- additional cross-scene datasets

---

# 3. Main architectural insight

BiDA collapses the spectral dimension into channels early and then mainly performs spatial semantic tokenization.

Our model should instead preserve an explicit latent spectral axis:

\[
X \rightarrow F \in \mathbb{R}^{N\times C\times B'\times H\times W}.
\]

Then build two complementary token sets:

\[
\boxed{\text{Spatial Semantic Tokens}}
\]

and

\[
\boxed{\text{Spectral Global Tokens}}.
\]

The intended interpretation is:

- spatial tokens: **where / spatial context**
- spectral tokens: **what spectral pattern**

These token sets are fused inside each HSI sample first, then used for source-target semantic interaction.

---

# 4. Module A — Spectral-Spatial Stem

Input:

\[
X\in\mathbb R^{N\times1\times B\times H\times W}.
\]

For Houston:

\[
B=48,\quad H=W=13.
\]

Use a shallow factorized 3D CNN stem.

Recommended V0 structure:

```text
Input HSI
  ↓
Spectral Conv3D: mainly k×1×1
  ↓
Spatial Conv3D: 1×3×3
  ↓
Spectral Conv3D: mainly k×1×1
  ↓
Latent feature cube F
```

Requirements:
- preserve an explicit spectral dimension
- target latent spectral length around \(B'=12\)
- do not collapse spectral dimension into channels
- use `GroupNorm + GELU`
- avoid BatchNorm in V0 to remove source-target mixed running-statistics effects

Suggested output:

\[
F\in\mathbb R^{N\times64\times12\times13\times13}.
\]

---

# 5. Module B — Spectral Global Tokens

Purpose:

> Aggregate global wavelength structure while preserving the order of latent spectral positions.

From:

\[
F\in\mathbb R^{N\times C\times B'\times H\times W},
\]

pool only spatial dimensions:

\[
f_b=\operatorname{GAP}_{H,W}(F[:,:,b,:,:]).
\]

This produces:

\[
X_{\text{spec}}\in\mathbb R^{N\times B'\times C}.
\]

Project to token dimension \(D\), then add a 1D spectral positional encoding:

\[
X_{\text{spec}}\in\mathbb R^{N\times B'\times D}.
\]

Use \(K_g\) learnable spectral queries:

\[
Q_g\in\mathbb R^{K_g\times D}.
\]

Cross-attention:

\[
Q=Q_g,\qquad K=V=X_{\text{spec}}.
\]

Softmax is over the \(B'\) spectral positions.

Output:

\[
G\in\mathbb R^{N\times K_g\times D}.
\]

Recommended V0:

\[
K_g=4.
\]

Interpretation:
- different spectral queries can learn different global spectral regions/patterns
- do not force one single global spectral summary token

---

# 6. Module C — Spatial Semantic Tokens

Purpose:

> Aggregate spatial/contextual semantics without destroying spectral information too early.

From:

\[
F\in\mathbb R^{N\times C\times B'\times H\times W},
\]

for every spatial position \((h,w)\), learn a spectral projection:

\[
x_{h,w}=P_{\text{spa}}(F[:,:,h,w]).
\]

This yields:

\[
X_{\text{spa}}\in\mathbb R^{N\times HW\times D}.
\]

For Houston:

\[
HW=169.
\]

Add 2D positional encoding.

Use \(K_s\) learnable spatial queries:

\[
Q_s\in\mathbb R^{K_s\times D}.
\]

Cross-attention:

\[
Q=Q_s,\qquad K=V=X_{\text{spa}}.
\]

Softmax is over the \(HW\) spatial positions.

Output:

\[
S\in\mathbb R^{N\times K_s\times D}.
\]

Recommended V0:

\[
K_s=4.
\]

---

# 7. Module D — Intra-Sample Spatial-Spectral Fusion

Before any source-target interaction, fuse the two token types inside the same HSI sample.

Use bidirectional cross-attention.

Spatial reads spectral:

\[
S'
=
S+
\operatorname{Attn}(Q=S,K=G,V=G).
\]

Spectral reads spatial:

\[
G'
=
G+
\operatorname{Attn}(Q=G,K=S,V=S).
\]

Recommended implementation:
- compute both directions in parallel from the same pre-fusion \(S,G\)
- Pre-LN
- residual connection
- lightweight MLP after attention
- no extra loss

The purpose is:

\[
\boxed{\text{Spatial context}\leftrightarrow\text{Spectral pattern}}
\]

within each sample.

---

# 8. Module E — Class Query for Final Prediction

Do not classify directly from a single spatial or spectral token.

After fusion:

\[
T_{\text{sem}}=[S';G'].
\]

Introduce one learnable class query:

\[
q_{\text{cls}}\in\mathbb R^D.
\]

Use:

\[
Q=q_{\text{cls}},\qquad K=V=T_{\text{sem}}.
\]

Obtain:

\[
z_{\text{cls}}
=
q_{\text{cls}}
+
\operatorname{Attn}(q_{\text{cls}},T_{\text{sem}},T_{\text{sem}}).
\]

Then:

\[
\hat y
=
\operatorname{Linear}(\operatorname{LN}(z_{\text{cls}})).
\]

The class query is the only token used for final classification.

---

# 9. Module F — Source Semantic Memory

Source-target interaction should not use arbitrary source-target batch pairing.

Use source labels to organize semantic memory.

For each class \(c\), maintain separate token-slot memories:

\[
M_c^{spa}
=
\{m^{spa}_{c,1},\dots,m^{spa}_{c,K_s}\},
\]

\[
M_c^{spec}
=
\{m^{spec}_{c,1},\dots,m^{spec}_{c,K_g}\}.
\]

Thus memory is indexed by:

\[
\boxed{\text{class} \times \text{token slot}}.
\]

Do not reduce each class to one global prototype.

Recommended V0 memory update:
- maintain EMA memory from labeled source samples
- update each class/token-slot using the corresponding source token
- no target pseudo-labels are used to construct the memory

For source sample \(i\) with label \(c\):

\[
m_{c,k}
\leftarrow
\mu m_{c,k}
+
(1-\mu)\bar t_{c,k}.
\]

Keep spatial and spectral memories separate.

---

# 10. Module G — Token-Aware, Class-Aware Source-Target Interaction

Only semantic tokens interact across domains.

Do not use domain/statistical tokens in source-target interaction.

For target spatial token slot \(k\):

\[
s^t_k
\]

query only the same spatial slot from all source classes:

\[
M_k^{spa}
=
[m^{spa}_{1,k},m^{spa}_{2,k},\dots,m^{spa}_{C,k}].
\]

Use:

\[
r^{spa}_k
=
\operatorname{Attn}
(
Q=s^t_k,
K=M_k^{spa},
V=M_k^{spa}
).
\]

The attention softmax is over source classes.

Similarly for spectral token slot \(k\):

\[
r^{spec}_k
=
\operatorname{Attn}
(
Q=g^t_k,
K=M_k^{spec},
V=M_k^{spec}
).
\]

This interaction is therefore:

\[
\boxed{\text{class-aware}}
\]

and

\[
\boxed{\text{token-aware}}.
\]

Target hard pseudo-labels are not required.

---

# 11. Module H — Source Information Injection

V0 should use simple residual injection rather than another learned sample-wise gate.

For spatial tokens:

\[
\tilde s^t_k
=
s^t_k
+
\alpha_{spa}r^{spa}_k.
\]

For spectral tokens:

\[
\tilde g^t_k
=
g^t_k
+
\alpha_{spec}r^{spec}_k.
\]

Use separate learnable scalars:

\[
\alpha_{spa},\quad\alpha_{spec}.
\]

Initialize them near zero:

\[
\alpha_{spa}\approx0,\qquad
\alpha_{spec}\approx0.
\]

Reason:
- training begins close to the target self representation
- source semantic information is introduced gradually
- avoids immediately overwriting target tokens

Do not add reliability/utility gates in V0.

---

# 12. Forward paths

## Source

```text
source patch
  ↓
shared spectral-spatial stem
  ↓
spatial tokens + spectral tokens
  ↓
intra-sample fusion
  ↓
class query
  ↓
source classifier
```

Source tokens also update class-aware semantic memory using source GT.

## Target

```text
target patch
  ↓
shared spectral-spatial stem
  ↓
spatial tokens + spectral tokens
  ↓
intra-sample fusion
  ↓
query source class/token memory
  ↓
zero-init residual semantic injection
  ↓
class query
  ↓
target prediction
```

---

# 13. Losses for V0

Keep the first version minimal.

Required source supervised loss:

\[
\mathcal L_{\text{src}}
=
CE(\hat y_s,y_s).
\]

Do not initially add:
- contrastive loss
- target pseudo-label CE
- entropy minimization
- prototype loss
- MMD / LMMD
- Flow Matching
- OT
- external priors

If the architecture alone does not show a useful signal, do not hide the failure by immediately stacking extra objectives.

A later V1 may add one carefully motivated semantic-only adaptation loss if V0 is promising.

---

# 14. Initial dimensions

Recommended starting values:

\[
C=64,\qquad D=128,
\]

\[
B'=12,
\]

\[
K_s=4,\qquad K_g=4.
\]

Attention:
- 4 heads
- Pre-LN
- GELU MLP
- small depth

These values are starting points, not target-GT-tuned hyperparameters.

---

# 15. Required implementation outputs

The model forward should expose intermediate representations for diagnostics:

```python
{
    "logits": ...,
    "z_cls": ...,
    "spatial_tokens": ...,
    "spectral_tokens": ...,
    "spatial_attn": ...,
    "spectral_attn": ...,
    "memory_spatial_attn": ...,
    "memory_spectral_attn": ...,
    "alpha_spatial": ...,
    "alpha_spectral": ...,
}
```

This is required for later causal analysis and visualization.

---

# 16. Minimum ablation sequence

Do not start with the full model only.

### A0 — stem baseline
```text
stem → pooling → classifier
```

### A1 — spatial tokenization
```text
stem → spatial tokens → class query
```

### A2 — dual-axis tokenization
```text
stem
→ spatial tokens + spectral global tokens
→ intra-sample fusion
→ class query
```

### A3 — full V0
```text
A2
+ source class/token semantic memory
+ target memory interaction
+ zero-init residual injection
```

This directly tests:

1. whether explicit spectral-global tokens help
2. whether class-aware/token-aware source memory helps beyond the representation itself

---

# 17. Diagnostics

Always report:
- OA
- AA
- Kappa
- per-class recall
- mean/std over seeds

Also inspect:
- spectral attention over latent wavelength bins
- spatial attention maps
- source-class attention for target spatial tokens
- source-class attention for target spectral tokens
- learned \(\alpha_{spa}\) and \(\alpha_{spec}\)
- percentage of target predictions changed by memory injection
- correct→wrong / wrong→correct changes as post-hoc diagnostics

Do not infer mechanism only from final OA.

---

# 18. Main research story for V0

The intended story is:

> Existing cross-scene HSI models can collapse spectral structure too early and then perform source-target interaction in a mixed token space. We instead preserve an explicit spectral axis, construct complementary spatial and spectral semantic tokens, and perform source-target interaction through source-label-organized, token-specific semantic memories.

The two central questions are:

\[
\boxed{\text{Does explicit dual-axis tokenization improve HSI semantic representation?}}
\]

and

\[
\boxed{\text{Does class-aware/token-aware source semantic interaction improve cross-scene transfer?}}
\]

V0 should answer these before any additional loss or module is introduced.
