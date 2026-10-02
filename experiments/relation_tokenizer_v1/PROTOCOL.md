# Houston relation tokenizer V1: frozen before target outcomes

Houston13 → Houston18; seeds 2100/2101/2102; all three arms freshly trained.
Fixed epoch 200 is the sole endpoint. Do not select an arm/checkpoint or append
experiments automatically from target metrics. No Pavia processes/files touched.

## Only intervention: spatial scoring

A: original BiDA conv_a 64→4 spatial scores, Full Joint BN.
B: per-band u=(x−patch_mean)/(patch_std+1e-6) from the preprocessed, augmented
13×13 input patch (48 bands). Mean/std over all 169 spatial positions; std is
population std (correction=0). Shared pointwise MLP Linear(48,32,bias=True),
ReLU, Linear(32,4,bias=True), then spatial softmax separately for each token.
C: identical MLP and initialization, r=(x−center)/(patch_std+1e-6), center=(6,6).
The 32-unit width is a fixed lightweight implementation choice, not a search.
Both use original CNN features as value (64 dimensions); both padded stride-1
convolutions retain the exact 13×13 grid. Assert exact grid alignment; no resizing.
No positional encoding, gate, center branch, extra readout or adaptation losses.
Remove unused conv_a in B/C; all other original trainable parameters remain.
Initialize the relation MLP in an isolated CPU RNG stream seed+41000 so original
model/dropout/data-order RNG is unchanged. B/C MLP initialization is identical.

## Locked historical training

Original BiDA stem, spatial tokenizer value, four tokens, dim64, depth3, dropout0.1,
Full Joint BN (`bn_stabilization_v1/normalization.py`, differentiable native pooled
source/target moments, momentum0.19), source-self CE only, SGD0.01 no momentum
or schedule, batch128/domain, 200 epochs, normband, reflect padding, source95%/5%
split random_state23, original augmentation and sampler. Instantiate original
model then unused EMA before intervention, as historically frozen. Target GT mask
defines candidate centers as historically; target class values enter neither
loss, model structure, checkpoint selection, early stopping nor tuning.

Save every10 epochs, infer only epoch200. All nine prediction files and checkpoint
hashes must be frozen before any new target scores are computed. Track batch and
pre-forward CPU/CUDA RNG hashes; compare every epoch with historical Full Joint
controls (`results/sceneshift_joint_v1/C_<seed>`). A must replay historical final
tensors, including BN buffers, exactly. No historical files overwritten.

## Implementation gate and reporting

Before full training: test B/C relation formulas, constant patches, spatial
softmax normalization, exact grid, unchanged shared tensors, equal B/C MLP
initialization, finite logits/loss/gradients, nonzero MLP gradients, and nonzero
target CNN activation/input gradients under source-only CE via joint moments.
Report fixed200 OA/AA/Kappa, source-val OA, per-class recall, prediction shares,
class6 collapse (≥95%) and any-class collapse, zero-recall classes, parameters,
training/inference time, paired C−A and C−B and mean±sample std(ddof1).
Retain configs, snapshots/code diff, checkpoints, manifest hashes and full logs.
Report all arms without automatically promoting a target-best version.
