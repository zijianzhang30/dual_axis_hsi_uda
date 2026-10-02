# Tokenizer x readout factorial: stress seed2101

Frozen before training new cells. Full Joint BN, zero BiDA refinement blocks,
tokenizer in {BiDA spatial-softmax, existing A1 query package}, readout in
{A1 Class Query CrossBlock, mean tokens}, all followed by the identical
LayerNorm/linear architecture. Four cells: query_class reused from completed
query_class_readout_v1/formal_2101; new bida_class, query_mean, bida_mean.
The extra bida_mean cell completes the requested2x2, allowing interaction checks.

Same seed2101, Houston13->Houston18, fixed split, normband, patch13, batch128
per domain, source CE only, SGD .01 without momentum/schedule, token dropout
.1 before readout, Full Joint native BN .19 per paired step with target moment
gradients. 200 epochs; no ratio, learning-rate, loss or checkpoint tuning.

Build original BiDA3 plus unused EMA and temporary Query transplant; verify
previous initial BiDA/Hybrid hashes. Initialize common A1 Class Query head,
verify exact existing query_class initial pipeline hash. Restore the originally
initialized conv_a/semantic method for BiDA cells and remove spatial-query
parameters; remove Class Query parameters for mean cells. Discarded modules
are initialized identically before removal, matching pre-training RNG state
across all cells and the reused control. Shared initial tensors must match.
Attention in both readers has native default zero dropout; BiDA blocks absent.
No added readout activation/projection: mean -> retained LayerNorm -> linear.

All cells keep four64-dim tokens. Token dropout calls have identical shapes
and order; reader modules introduce no extra stochastic dropout. Parameters
and learned activation distributions differ. BiDA versus Query tokenizer is
a package intervention including query, positional encoding, attention/FFN,
not an isolated learnable-query parameter comparison.

Save10:10:200 and freeze/hash all20 predictions before post-hoc GT metrics.
Fixed200 primary; target-oracle best OA separately labeled diagnostic only.
Report OA/AA/Kappa, recall, prediction shares, entropy, source-val and class6
collapse >=95% both endpoint and trajectory. Single seed, no cross-seed SD.
Diagnostic improvement must be assessed with AA/coverage as well as OA;
not crossing95% is not evidence of full stability.

Use the same fixed2048 unlabeled source/target patches as previous probes
for frozen eval and current joint-training moments, dropout disabled. Record
token mean and classifier z norms, direction concentration, local w6 cosine,
class6 margins/shares. No gradients/weights updated; restore and hash-check
state after joint-mode probes. Retrospective snapshots, not historical traces.

Factorial differences are descriptive at this optimization seed. Opposite
readout/tokenizer effects would indicate interaction; no pattern uniquely
proves the spatial reader is intrinsically wrong or generalizes to other seeds.
Do not add rescue losses or change main Full Joint+BiDA3 reference.
