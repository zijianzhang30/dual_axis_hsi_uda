# BiDA-self tokenizer transplant: seeds 2100/2101/2102

Frozen before training or inspecting new target outcomes.

## Question and arms

Control is paired BiDA-self with its 1x1-convolution spatial softmax
tokenizer, mixed source/target BN, depth 3, dropout 0.1, and source CE.
Replace only its four-token extractor with the complete A1 spatial
learnable-query tokenizer: four learned 64-dimensional queries, two-dimensional
sinusoidal pixel positions, and A1 `CrossBlock` (four attention heads, residual
query, and feed-forward layer). Feed its four outputs into the same BiDA CLS
token, positional embedding, three blocks, normalization, and classifier.
The new tokenizer operates directly on the same 64-channel BiDA stem feature
map; A1's stem and class reader are excluded. BiDA's old `conv_a` is removed.

Instantiate and hash the complete BiDA model before surgery. Preserve every
shared initial tensor exactly. Instantiate the unused EMA control before the
new tokenizer to preserve the control's pre-intervention RNG sequence. Report
new parameter count because the tokenizer modules have different sizes.

## Fixed protocol

Houston13 to Houston18, `normband`, 13x13 patches, batch 128, source split
`random_state=23`, seeds 2100/2101/2102, SGD 0.01 without momentum or schedule, 200
epochs, paired source/target forward for mixed BN, source self CE only, target
self inference. Keep dropout 0.1 and depth 3 in both arms. Fixed epoch 200 is
the primary target-blind endpoint.

Save checkpoints every ten epochs. After training, freeze and hash target
predictions for all 20 checkpoints before reading target labels. Report the
best target OA on this grid as a separately labeled target-oracle training
trajectory diagnostic. It must not select the formal checkpoint or tune the
architecture. Target GT values enter neither loss nor checkpoint selection;
the released GT mask defines eligible target centers.

## Interpretation

Report transplant minus BiDA-tokenizer OA, AA, Kappa, per-class recall,
prediction distribution, source validation, parameter count, and cost.
Report paired differences and mean plus sample standard deviation across the
three seeds. Treat the transplant as a stable improvement only if it exceeds
the paired BiDA-tokenizer control in all three seeds, its mean OA improvement
is at least two points, and mean AA moves in the same direction. Otherwise the
seed-2100 gain is not sufficient to promote the Hybrid as the new backbone.
