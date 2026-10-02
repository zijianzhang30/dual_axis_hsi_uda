# BiDA-self tokenizer transplant: three-seed result

Houston13 to Houston18, seeds 2100/2101/2102, fixed epoch 200. The A1 spatial
query tokenizer was transplanted onto the BiDA stem. Both arms use mixed
source/target BN, three BiDA blocks, dropout 0.1, source self CE, and target
self inference. Each seed has a paired BiDA spatial-softmax tokenizer control.

| Seed | Control OA | Hybrid OA | OA change | Control AA | Hybrid AA | AA change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 78.41% | **81.71%** | **+3.30** | 66.32% | **68.00%** | **+1.68** |
| 2101 | **63.72%** | 62.59% | **-1.13** | **32.82%** | 26.38% | **-6.44** |
| 2102 | 77.40% | **78.15%** | **+0.74** | **63.01%** | 62.15% | **-0.86** |
| Mean ± sample SD | 73.18 ± 8.21% | 74.15 ± 10.17% | +0.97 ± 2.22 | 54.05 ± 18.46% | 52.18 ± 22.53% | -1.87 ± 4.15 |

The transplant does not pass the preregistered promotion gate. It improves OA
in two rather than all three seeds, mean OA improves by only 0.97 points, and
mean AA decreases by 1.87 points. It also increases OA and AA variance. The
seed-2100 gain is a positive single-seed signal rather than a stable tokenizer
advantage, so the Hybrid is not promoted to the main backbone.

The intervention replaces the complete tokenizer mechanism: four learned
queries, two-dimensional sinusoidal positions, four-head cross attention,
residual path, and FFN. It adds 33,600 parameters. Query, position encoding,
FFN, and capacity therefore remain confounded, but their decomposition is
deferred because the complete module did not survive the stability test.

All Hybrid ten-epoch predictions were frozen before target metrics. The
target-oracle diagnostics are 82.69% at epoch 130 for seed 2100, 76.79% at
epoch 10 for seed 2101, and 79.26% at epoch 120 for seed 2102. They describe
trajectory instability and do not replace fixed epoch 200. In seed 2101,
class-6 prediction share rises from 71.20% at epoch 10 to 95.90% at epoch 20
and ends at 98.19%.

The dedicated BN audit shows that this failure is strongly controlled by
normalization state; see
`experiments/mixed_bn_stability/RESULTS.md`. The next priority is stabilizing
mixed BN across seeds before further tokenizer or UDA expansion.

Artifact directories:

- `results/bida_self_tokenizer/diagnostic_2100/`
- `results/bida_self_tokenizer/formal_2101/`
- `results/bida_self_tokenizer/formal_2102/`

Seed 2101 encountered a full filesystem while writing its epoch-200 prediction.
The incomplete unhashed file was removed, and that prediction was regenerated
from the already frozen epoch-200 checkpoint without retraining. All 20
checkpoint and prediction hashes were then verified. Target labels were used
only after predictions were frozen.
