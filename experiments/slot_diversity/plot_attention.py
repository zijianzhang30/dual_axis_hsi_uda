"""Save comparable mean spatial attention maps for each paired seed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2100, 2101, 2102), required=True)
    args = parser.parse_args()
    rows = []
    for arm, domain in (("d0", "source_val_at_final"), ("d1", "source_val_at_final"),
                        ("d0", "target_at_final"), ("d1", "target_at_final")):
        path = ROOT / "results" / "slot_diversity" / f"{arm}_{args.seed}" / "summary.json"
        summary = json.loads(path.read_text())
        maps = np.asarray(summary[domain]["mean_spatial_attn"])
        side = int(np.sqrt(maps.shape[1]))
        if side * side != maps.shape[1]:
            raise ValueError("Spatial attention does not form a square map")
        rows.append((f"{arm.upper()} {'source val' if domain.startswith('source') else 'target'}",
                     maps.reshape(maps.shape[0], side, side)))
    vmax = max(float(maps.max()) for _, maps in rows)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    for row_id, (name, maps) in enumerate(rows):
        for slot in range(4):
            ax = axes[row_id, slot]
            image = ax.imshow(maps[slot], cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"{name}, slot {slot + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.65,
                 label="Mean attention weight")
    output = ROOT / "results" / "slot_diversity" / f"attention_maps_{args.seed}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
