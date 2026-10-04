# Full BiDA under TSTnet-style HyRANK protocol

One additional task only: arm A, seed1233, Dioni -> Loukia, 500 epochs.
Output: `/nas1/zhangzj26/dual_axis_hsi_uda/results/tstnet_pipeline_bida_v1`.
Do not change/stop current native TSTnet or B/C tasks.

Reuse B/C frozen TSTnet-style data setup, sampling, augmentation, optimizer,
loader and evaluation logic. Source 5% split random_state23, repeat10,
L2 norm, patch12 symmetric padding, batch100, 100 steps/epoch.
Same target GT-stratified subset and native augmented test mask. Target
labels replaced by zeros before optimization; GT influences native sampling
and evaluation but never enters any loss. This is a protocol diagnosis.

Full official BiDA architecture: original tokenizer, depth3/dim64/K4/dropout.1,
Original Mixed BN momentum.1, independent student/EMA initialization.
NO Joint BN or relation tokenizer. Retain complete published inner-loop AST
without changing losses or EMA semantics, including teacher not detached,
double-softmax consistency, second MMD self-pair, parameters-only EMA.
lambda1=.1, lambda2=1, ema_decay=.999, label smoothing off.
Native make_loss call and its unused CenterLoss construction retained.
MMD starts epoch101 as in released code; do NOT proportionally move it to251.

Deliberately changed optimizer/budget to match TSTnet-style B/C:
SGD recreated every epoch, momentum.9, decay1e-4,
lr=.001/(1+10*zero_based_epoch/500)^.75, 500 epochs/50,000 updates.
Do not call this unmodified official BiDA training or paper reproduction.

Primary fixed500; native-grid1,11,...,491 target-best only diagnostic.
Terminal fixed500 evaluation occurs after all training. No early stopping,
target-based tuning, extra source validation or automatic added seeds.
Save student and EMA final states, logs, predictions, metrics and full adapter
diff. Compare matched epoch A/B/C, not unrelated per-run best checkpoints.

Preflight separately executes actual native full-batch100 epoch100/101,
checks finite losses/gradients and MMD branch off/on. No preflight RNG/state
is reused in training. Existing B/C scripts are referenced but not edited.
Archive full BiDA dependencies in a separate directory, not active originals.
Inherited modern torch/PyG limitations recorded by TSTnet native run.
