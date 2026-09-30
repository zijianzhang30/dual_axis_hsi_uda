# BiDA-self transformer-depth decomposition: seed 2100

Frozen before training or inspecting new target outcomes.

## Question

After retaining the necessary mixed source/target BatchNorm path, how much of
BiDA-self's cross-scene performance comes from three transformer refinement
blocks rather than one?

## Intervention

- Control: completed BiDA-self with depth 3, mixed BN, dropout 0.1, source CE.
- Intervention: identical setup with only the first transformer block retained.

Construct the same initialized depth-3 model and unused EMA model as the
control, verify the initial model hash, then remove blocks 2 and 3. This keeps
the stem, tokenizer, position/CLS tokens, first block, normalization, and
classifier exactly initialized as in the depth-3 control. Target batches remain
in every forward and update mixed BN statistics.

## Fixed protocol

Houston13 → Houston18, `normband`, 13×13 patches, batch 128, source split
`random_state=23`, seed 2100, SGD 0.01 without momentum or schedule, dropout
0.1, source CE only, 200 epochs, fixed epoch-200 target self inference. Target
labels enter neither loss nor checkpoint selection; the released target GT
mask still defines eligible centers.

Predictions and hashes are frozen before post-hoc target metrics.

## Interpretation

Report depth-3 minus depth-1 OA, AA, Kappa, per-class recall, prediction
distribution, entropy, parameter count, source validation, and cost. A depth-3
advantage of at least 2 OA points with AA moving in the same direction is a
material refinement-depth effect for this seed. If depth 1 is within 2 OA
points of depth 3 without collapse, three-block refinement is not required to
explain the seed-2100 result. No depth or learning-rate search is allowed.
