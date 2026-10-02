# Frozen BiDA official-benchmark validation V1

All primary results use epoch200, seeds2100/2101/2102. No target-based tuning,
model/checkpoint selection, early stopping, or additional experiments. No changes
to Pavia, existing source code or historical artifacts. New outputs go to NAS,
with an engineering-project symlink; root disk has only ~1 GB available.

## Dataset status and provenance

HyRANK is the first runnable task. Downloaded from the author's Data-CSHSI
Google Drive folder13T47hw6RZnz1_CE7gG1QgYIYS2SDJ5DD into
`/nas1/zhangzj26/dual_axis_hsi_uda/datasets/HyRANK_author_v1/HyRANK/`.
Use ONLY Dioni_gt_out68.mat and Loukia_gt_out68.mat, key map. Images use ori_data.
Author README explicitly identifies these screened labels. Dioni250×1376×176,
Loukia249×945×176, 12 mapped classes. Raw *_gt.mat files are not used.
Reference: https://github.com/YuxiangZhang-BIT/Data-CSHSI

Load with the actual released `utils.dataset.load_mat_hsi` Dioni/Loukia branches.
The actual released main.py calls normband for both names; use that operation,
not Houston-specific spectral slicing, PCA, ILDA, z-scoring or new remapping.
The released factory explicitly registers 176 bands and12 classes. Preserve its
HSIDataset boundary rule (first image row/column excluded after reflect padding);
record the difference between raw GT mask counts and usable evaluation centers.

Official explicit generic main.py defaults: source95% stratified split,
sample_gt random_state23, re_ratio1, patch13, bs128, dim64, depth3, tokens4,
SGD.01, lambda1.1, lambda2=1, EMA.999, 200epochs. The repository has no separately
documented HyRANK-specific overrides: do not claim these are paper-tuned HyRANK
values. Uniform inherited frozen settings: dropout.1, source-val monitoring at
epoch1 and every10 epochs, fixed200 primary, independent post-training scoring.
Record this distinction in each dataset manifest. Candidate training/test centers
use author-filtered target GT nonzero mask, not all target pixels. Actual target
class values are replaced by dummy zeros in training and never enter losses.
Evaluation uses the same mask and dataset boundary rule, after frozen predictions.

MFF-SD→TD1/TD2: blocked pending original dataset and verified SD/TD1/TD2 split.
No MFF files found under the user's home or NAS. The author's public data page
does not supply MFF. An MJG factory hint (64bands/5classes) is NOT sufficient
evidence of MFF naming, regions, class mapping or split; do not substitute data.

Houston: B/C epoch200 artifacts are audited for reuse. Historical strict A uses
an experimental runner with a reordered post-epoch100 loss expression and
different evaluation cadence, rather than executing the released training loop.
Do not silently label that old checkpoint an exact released-loop reproduction.
Reuse eligibility and any blocked A result are recorded, not replaced without
reporting. No new Houston training in the initial HyRANK-priority launch.

## Three arms

A: actual released BiDA model, native sequential BN, all branches, native EMA
teacher, classification+bidirectional distillation+source/target consistency,
and native MMD after epoch100. Compile the released train() AST unchanged except
its target-evaluation/target-best-saving block is replaced by a target-blind
callback for logs, source validation and fixed checkpoint saving. Preserve the
exact native loss expressions, term order, batch-size conditions, global-step
EMA update and the released (possibly unusual) MMD self-self term. Do not fix or
reinterpret it. Original training-loop diff is saved. Target labels yielded to
that native loop are dummy zeros. Fixed endpoint differs from paper/code oracle.

B: same initialized BiDA, differentiable Full Joint BN(momentum.19), original
tokenizer, source-self CE only. Instantiate then discard the same EMA model.
C: B with unmodified current center-relative relation tokenizer. Adapt only its
input band literals48→176 (constructor, shape validation, error string), saving
the derived source and diff. Same eps1e-6, population patch std, center(6,6),
MLP D→32→4/ReLU, spatial softmax; CNN value dim64/grid13, all other modules fixed.
Isolated MLP RNG seed+41000 as historically frozen. No new network components.

All arms share data masks, split, per-seed initialization of common components,
loader RNG/augmentation and source validation cadence. Full A consumes extra
EMA dropout randomness; that is an intended property of the released A, not
forced to match self-only gradients/RNG. B/C training batch and RNG streams must
match exactly. All three batch streams must match. Save checkpoints every10;
infer only epoch200 after training. Freeze all nine predictions before scoring.

## Gate and report

Inspect all required files/hashes/keys/bands/class maps/masks. Validate native A
AST transformation, original branch/BN types, logits/loss/finite gradients,
grid alignment, unchanged common tensors and nonzero target-joint-statistics
gradient in B/C. Start all nine only after checks pass. Report all seeds and
mean±sampleSD, OA/AA/Kappa, recall, prediction shares, any-class≥95% collapse,
zero-recall classes, source-val, parameters and wall time; paired C−B/C−A.
Paper values are a separate table with source/selection protocol. Never mix a
paper or target-oracle number into the local fixed200 table; unverified paper
entries remain unavailable, not guessed. No automatic version promotion.
