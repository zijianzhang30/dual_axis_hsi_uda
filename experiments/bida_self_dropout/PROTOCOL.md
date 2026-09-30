# BiDA-self dropout decomposition: seed 2100

Frozen before training or inspecting new target outcomes.

## Question

With mixed BN and three transformer blocks retained, how much of BiDA-self's
cross-scene performance comes from dropout regularization?

## Intervention

- Control: completed BiDA-self with dropout 0.1.
- Intervention: set every `torch.nn.Dropout` probability in the same initialized
  model to zero. Architecture, parameters, depth, tokenizer, mixed BN, target
  forward, optimizer, and source CE remain unchanged.

The full initialized model hash must match the completed seed-2100 control
before dropout probabilities are changed. Dropout has no trainable tensors, so
all initial parameter values remain identical.

## Fixed protocol

Houston13 → Houston18, `normband`, 13×13 patches, batch 128, source split
`random_state=23`, seed 2100, SGD 0.01 without momentum or schedule, depth 3,
mixed source/target BN, source CE only, 200 epochs, fixed epoch-200 target self
inference. Target labels enter neither loss nor checkpoint selection.

Save checkpoints every 10 epochs to match the released BiDA evaluation grid.
After all 200 training epochs, generate and hash target predictions for all 20
checkpoints before collecting target labels. Fixed epoch 200 remains the
primary target-blind endpoint. The maximum target OA over epochs 10, 20, ...,
200 is a separately labeled released-code-style target-oracle diagnostic and
must not select the formal checkpoint or guide hyperparameters.

## Interpretation

Report dropout-0.1 minus dropout-0 for OA, AA, Kappa, per-class recall,
prediction distribution, entropy, source validation, and cost. An absolute OA
difference of at least 2 points with AA moving in the same direction is a
material dropout effect for this seed. No dropout-rate search is allowed.
