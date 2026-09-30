# BiDA-self tokenizer transplant: seed 2100

The A1 spatial learnable-query tokenizer was transplanted onto the BiDA stem.
Both arms used mixed source/target BN, three BiDA transformer blocks, dropout
0.1, source self CE, and fixed epoch-200 target self inference. The complete
BiDA model matched the seed-2100 initialization hash before the intervention;
all 55 shared initial state tensors remained identical after the replacement.

| Tokenizer | Parameters | Target OA | AA | Kappa | Epoch-200 source-val OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 query + position + CrossBlock | 410,167 | **81.71%** | **68.00%** | **0.675** | **85.04%** |
| BiDA spatial softmax | 376,567 | 78.41% | 66.32% | 0.627 | 79.53% |

The transplant improves fixed-endpoint target OA by **3.30 points**, AA by
**1.68 points**, and Kappa by **0.048**. OA exceeds the predeclared two-point
material-effect threshold and AA moves in the same direction. This result
rejects the idea that BiDA's original tokenizer supplies a necessary second
advantage in this setting. It is consistent with the larger mixed-BN effect
being the dominant source of BiDA-self's performance, while a richer query
tokenizer can improve it further.

The intervention replaces the complete tokenizer mechanism, including A1's
two-dimensional sinusoidal positions, four learned queries, four-head cross
attention, residual path, and feed-forward layer. It adds **33,600** trainable
parameters. Therefore the result does not isolate query weighting from
position encoding, extra capacity, or the CrossBlock; those components would
require further ablations. The BiDA stem, CLS path, three refinement blocks,
classifier, optimization, and mixed-BN source/target ordering were held fixed.

At epoch 200, recall improves most for class 7 (+17.16 points), class 6
(+3.99), and class 1 (+3.47). It falls for class 5 (-8.77) and class 2
(-4.20). All seven classes retain nonzero recall. Prediction entropy falls
from 0.145 to 0.119, while class-6 prediction share increases from 63.66%
to 66.66%.

All 20 ten-epoch checkpoints and target predictions were frozen before target
labels were read. The separately labeled target-oracle diagnostic peaks at
epoch 130: **82.69% OA**, **70.66% AA**, and **0.703 Kappa**. It is 0.98 OA
points above epoch 200; fixed epoch 200 remains the primary result. The
control's completed run did not save a corresponding trajectory, so no
oracle-to-oracle comparison is made. Source validation's best value is 85.83%
at both epoch 20 and epoch 130; their target OAs differ by 1.96 points.

Training took 155.91 seconds and inference over 20 checkpoints took 66.15
seconds on GPU 4. Artifacts are under
`results/bida_self_tokenizer/diagnostic_2100/`.

- Frozen manifest SHA256:
  `7cf445ee293b592074af4c804a6e4b3cd8d352ad1afc8dc1e9d724c4236cc6bd`
- Epoch-200 checkpoint SHA256:
  `7734518fe23714a695f44ff55387fc473d6900bb4939963d00a6ddcbeabea684`
- Epoch-200 prediction SHA256:
  `384310b679dab4d576ccde9a5d713e622f77a2c601b0f00fa094b9dc227a0c04`
- A1 `model.py` source SHA256 at run time:
  `f22272f929fb5e112ab8d1a85adf3e6e1bfbdb096efe5551dd17bc977843644f`

This is one diagnostic seed. Target labels were used only after all prediction
artifacts and hashes had been frozen. Multi-seed validation is needed for a
general claim.
