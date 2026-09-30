# Dual-axis HSI UDA

Independent HSI cross-scene model project using the Houston data pipeline from `IEEE_TCSVT_BiDA`. The initial study follows the [revised V0 protocol](docs/V0_REVISED_PROTOCOL.md) and tests A0–A3 under a fixed epoch-200 protocol.

| Path | Purpose |
| --- | --- |
| `model.py`, `data.py`, `train.py` | Current working model and training entry |
| [`experiments/v0/`](experiments/v0/README.md) | Independent V0 code snapshot, protocol, controller, and aggregation |
| `investigations/` | Isolated diagnostics and follow-up analyses |
| `docs/` | Design and protocol documents |
| `tests/` | Fast model and data contracts |
| `results/` | Local outputs and checkpoints, excluded from version control |

From the project root, run one study arm with the existing environment:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/v0/train.py --variant a3 --seed 2100 --out results/v0/a3_2100
```

Or run the A0–A3 × three-seed matrix:

```bash
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python experiments/v0/run_formal.py --device cuda:0
```

The controller refuses to overwrite existing runs. See [V0 experiment notes](experiments/v0/README.md) for implementation details and reporting.
