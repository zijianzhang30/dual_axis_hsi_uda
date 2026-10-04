# Our models under TSTnet-style HyRANK protocol: single-seed diagnosis

Independent output: `/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_self_v1`.
Do not change or stop the parallel full TSTnet native reproduction.

## Models

B: BiDA-self + Full Joint BN + original spatial tokenizer.
C: same backbone and Full Joint BN + frozen center-relative relation tokenizer.
Both use ONLY source CE. No TSTnet graph branch, GCN CE, GOT, MMD,
consistency loss, distillation, EMA training or target pseudo-labels.
Historical unused EMA construction is retained only for initial RNG parity.

Frozen models: depth 3, dim 64, four tokens, dropout 0.1, Joint BN momentum
0.19. Adapt input to 176 bands and classifier to 12 classes. Relation MLP is
176 -> 32 -> 4 with ReLU, spatial softmax, eps=1e-6, original CNN values.

Patch 12 is even: retain frozen center indexing H//2,W//2 = (6,6), which is
the labeled position selected by the TSTnet loader. Do not change to 13 or
introduce a new pooling/readout. Preflight asserts relation weights (N,4,144),
aligned CNN grid, finite gradients and nonzero target-input gradient from CE.

## Data and training protocol

Reuse the pinned TSTnet native train.py MAIN data setup verbatim via AST,
with only model factory and train/test function adapters substituted.
Native snapshot is at `results/tstnet_hyrank_native_v1/run_snapshot` on NAS.
Its official upstream commit is f663d42bbb63fad126a86e6b259ef70412715ca2.
Previously recorded MAT reading and np.bool compatibility patches retained.

Use author HyRANK out68 labels, native per-pixel L2, symmetric padding,
5% stratified source split random_state=23, source repeat 10, batch 100.
Target candidates are the native GT-stratified subset at the native ratio;
target class values affect this sampling protocol but NEVER enter loss.
Explicitly NOT a fully target-label-blind sampling experiment.

Seed 1233, one trial per arm, 500 epochs. Same source/target datasets, shared
native loader generator, native shuffle, augmentation and prefetch order.
Source controls epoch length, native target iterator restart condition kept,
100 steps per epoch / 50,000 total updates. Keep flip/radiation augmentation.
Native SGD recreated each epoch: lr=.001/(1+10*epoch/500)^.75, momentum .9,
weight decay 1e-4. This changes the protocol, not the frozen network design.

## Evaluation and interpretation

Native evaluation grid 1,11,...,491, including native test augmentation.
Use our target self output; derive OA/AA/Kappa, all recalls, confusion matrix,
class shares, single_class_collapse>=95% and zero recall classes from saved
predictions. Retain native report/checkpoint calls; native target-best is only
a separately labeled diagnostic. No target-based early stopping or tuning.

Save final-500 state, then evaluate it once AFTER training completes. This
extra terminal evaluation does not alter a future training data/RNG stream.
Fixed epoch500 is primary; never replace it with target-best. Report C-B.
No source validation added: TSTnet uses source 5%, not our historical 95/5.

No claim of pure model comparison to old 200-epoch HyRANK results: data,
augmentation, sample budget and optimizer all differ. Goal is a protocol
diagnosis, not guaranteed recovery to 60%, and not a final multi-seed result.
Do not auto-expand to more seeds, tune to target accuracy or add losses.
Modern environment limitations are inherited from TSTnet reproduction.
