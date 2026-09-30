"""Inspect spatial-slot separation in the completed A1/B0 checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from train import ROOT, evaluate, set_seed
from experiments.v0.data import source_loaders, target_loader
from experiments.v0.model import DualAxisModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2100, 2101, 2102), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    set_seed(args.seed)
    _, source_val, classes, bands = source_loaders(
        seed=args.seed, num_workers=args.num_workers)
    target, target_classes = target_loader(seed=args.seed, num_workers=args.num_workers)
    if classes != target_classes:
        raise ValueError("Source and target class counts differ")
    model = DualAxisModel(bands=bands, classes=classes, variant="a1").to(args.device)
    checkpoint = ROOT / "results" / "v0" / f"a1_{args.seed}" / "final.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False)["model"])
    result = {"seed": args.seed, "checkpoint": str(checkpoint),
              "source_val": evaluate(model, source_val, args.device, classes, diagnostics=True),
              "target": evaluate(model, target, args.device, classes, diagnostics=True)}
    output = ROOT / "results" / "slot_uda" / f"b0_diagnostics_{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for domain in ("source_val", "target"):
        cosine = torch.tensor(result[domain]["mean_slot_cosine"])
        attention = torch.tensor(result[domain]["mean_attention_cosine"])
        upper = torch.triu_indices(len(cosine), len(cosine), offset=1)
        print(json.dumps({"seed": args.seed, "domain": domain,
                          "mean_slot_cosine_offdiag": cosine[upper[0], upper[1]].mean().item(),
                          "mean_attention_cosine_offdiag": attention[upper[0], upper[1]].mean().item(),
                          "target_oa": result[domain]["oa_percent"] if domain == "target" else None,
                          "output": str(output)}))


if __name__ == "__main__":
    main()
