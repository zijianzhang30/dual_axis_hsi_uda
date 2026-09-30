"""Post-hoc attention maps for one target example from each GT class."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v0.data import target_loader
from experiments.v0.model import DualAxisModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("d0", "d1"), required=True)
    parser.add_argument("--seed", type=int, choices=(2100, 2101, 2102), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    loader, classes = target_loader(seed=args.seed, batch_size=128, num_workers=0)
    dataset = loader.dataset
    labels = np.asarray(dataset.labels)
    examples = [int(np.flatnonzero(labels == class_id)[0]) for class_id in range(classes)]
    images = torch.stack([dataset[index][0] for index in examples]).to(args.device)
    model = DualAxisModel(bands=images.shape[2], classes=classes,
                          patch_size=images.shape[-1], variant="a1").to(args.device)
    folder = ROOT / "results" / "slot_diversity" / f"{args.arm}_{args.seed}"
    model.load_state_dict(torch.load(folder / "final.pth", map_location=args.device,
                                     weights_only=False)["model"])
    model.eval()
    with torch.inference_mode():
        output = model(images)
    attention = output["spatial_attn"].cpu().numpy().reshape(classes, 4, 13, 13)
    predicted = output["logits_self"].argmax(1).cpu().numpy()
    fig, axes = plt.subplots(classes, 4, figsize=(9, 2 * classes), constrained_layout=True)
    for class_id in range(classes):
        for slot in range(4):
            ax = axes[class_id, slot]
            image = ax.imshow(attention[class_id, slot], cmap="magma", vmin=0,
                             vmax=attention.max())
            ax.set_title(f"GT {class_id + 1}, pred {predicted[class_id] + 1}, slot {slot + 1}",
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.55,
                 label="Attention weight")
    path = ROOT / "results" / "slot_diversity" / f"class_examples_{args.arm}_{args.seed}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    metadata = {"seed": args.seed, "arm": args.arm, "example_indices": examples,
                "true_classes_1based": list(range(1, classes + 1)),
                "predicted_classes_1based": (predicted + 1).tolist(),
                "purpose": "post-hoc target GT class sampling for diagnostic visualization only"}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
