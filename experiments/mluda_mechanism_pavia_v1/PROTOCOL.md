# Full MLUDA mechanisms: PaviaU → PaviaC only

Independent four-arm study; existing BiDA/HyRANK files and tasks are untouched.
M0 native full MLUDA; M1 Joint CNN BN; M2 center-relative relation pooling;
M3 both. Fixed first three native seeds: 1622, 1322, 1256. No accuracy selection.
Native frozen snapshot/environment from completed mluda_pavia_native_v1.

Preserve whole-cube z-score + ILDA, exact native 180/class samples, patch11,
batch32/drop_last, 100 epochs/38 updates per epoch, radiation/flip augmentation,
source CE + 0.3*lambda*LMMD + both contrastive losses, constant lr=.001 SGD
momentum=.9/wd=.0005 recreated EACH epoch and all original parameter groups.
Keep native unused loss computations. No SceneShift or extra loss.

Native flip uses np.fliplr/flipud on B×C×H×W (channel/batch reversal), not H/W.
Preserve it, including labels/pairing behavior; do not silently repair it.
Relation always uses the actual preprocessed/augmented patch seen by CNN.
Center=(5,5), unchanged 11×11 grid; (patch-center)/(population spatial std+1e-6),
shared Linear102→32→4/ReLU, four spatial softmaxes, post-CA/SA CNN values,
four 288-D weighted vectors averaged to 288-D. Original cross-attention retained.
New parameters live under feature_layers and use independent seed+41000 CPU RNG.

Only executed CNN BN1–7 change: concatenate S/T separately per original/noise/flip
forward; split CNN values before native CA/SA calls. BN9/10 are not executed.
Native .1 momentum updates twice per pair; joint .19 updates once: matching
old-statistics decay (.9²=.81), NOT equivalent batch moments/normalization.
Eval uses running stats and the original final-source-batch target pairing.

Recover historical pre-ILDA RNG from initial_rng.pt and cache one ILDA result.
Record hashes; identical preprocessing/sampling/batches/augmentation/RNG across
arms is mandatory. Historical first-three checkpoints and stream artifacts are
absent, so strict reuse cannot be established: rebuild all three M0 controls.
Compare their scores to historical individual runs, never to the ten-seed mean.

Before training: disabled switches exactly match native outputs/gradients/updates;
uniform relation weights reproduce avgpool within numerical tolerance; validate
BN calls, target gradients through joint statistics, pooling gradients/updates,
parameter groups, initialization/RNG invariance, center/grid and no cross-view mix.
Freeze all 12 final predictions/checkpoints before target scoring. No target
metrics during training, checkpoint search, tuning, automatic additional seeds,
retry, or follow-up training. Native OA uses full target dataset denominator;
AA/Kappa use only retained predictions. Preserve and disclose this difference.
Collapse=max predicted share≥.95; also report zero-recall classes.

Report all endpoints, recall/shares/confusions, parameters/time and paired
M1−M0, M2−M0, M3−M1, M3−M2. Both population/sample SD; paired sample SD.
NAS new immutable directory mluda_mechanism_pavia_v1; refuse overwrites.
