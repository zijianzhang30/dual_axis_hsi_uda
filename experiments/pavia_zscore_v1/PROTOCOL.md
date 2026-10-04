# Pavia normalization diagnostic: seed 2100 only

User-authorized mechanism diagnostic, not MLUDA reproduction and not a new
preregistered generalization claim. Exactly two new runs: B original tokenizer
and C frozen center-relative tokenizer, both BiDA-self + Full Joint BN.
Only input normalization changes from spectral L2 normband to MLUDA's
per-scene, per-band sklearn.preprocessing.scale on the whole raw image.
Include unlabeled/background pixels in normalization just as the official
load_data_pavia function does. Do not first L2-normalize, add ILDA/PCA, or clamp.
HSIDataset retains its native float32 patch conversion.

PaviaU -> PaviaC, same existing seven-class mappings and 102 bands. Seed2100,
13x13 reflect patches, 95/5 stratified source split with random_state23,
batch128, SGD0.01 without momentum/weight decay/scheduler, 200 epochs,
depth3/dim64/K4/dropout0.1, joint BN momentum0.19, source CE only.
Retain the historical unused second model initialization and relation MLP
isolated initialization seed+41000. Retain source validation at epoch1 and
every10 epochs and zip(source,target) with unequal-batch skipping.
No class rebalancing, new objective, SceneShift, gate or pseudo-label.

Reuse B normband seed2100 from pavia_joint_v1/joint_2100 and C normband
seed2100 from mluda_bida_extension_v1/pavia/C_2100. Preserve all original
training processes, source files and results. B/C normalization variants must
share initialization and consumed sample/augmentation order. Audit this from
actual consumed batches: append a normband shadow image as extra dataset
channels, apply ONE native spatial augmentation to all channels together, and
split before model input. The shadow channels NEVER enter the model. This
adds no RNG calls or loader iterations. Hash shadow patches and source labels
against historical normband batch hashes at every epoch, and compare native
pre-forward CPU/CUDA RNG hashes. Also record actual dataset-index hashes.
Fail on a mismatch; no automatic restart or new experiments.

Preflight: compare normalization numerically with the actual isolated official
MLUDA load_data_pavia function; check data/GT hashes, indices, initialization,
finite tensors/gradients, relation softmax/grid/formula, and source CE target
input gradient through Joint BN. Preflight is separate from training.

Save every10 epochs, infer target only after200. Freeze and hash both new
prediction files before target scoring. Fixed200 only; no oracle, early stop,
target-guided tuning or version selection. Report single-seed OA/AA/Kappa,
per-class recall, prediction shares, single_class_collapse >=95%, zero-recall
classes, source-val, parameters/time, and zscore-minus-normband per tokenizer.
No mean/std or stability claims from one seed. Results go to a new NAS folder.
