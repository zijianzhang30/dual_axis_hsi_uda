"""Reuse the Strict BiDA Houston loader and its exact source split conventions."""

import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_BIDA_ROOT = Path("/home/zhangzj26/IEEE_TCSVT_BiDA")
DEFAULT_DATA_DIR = Path("/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston")


def _bida_dataset(bida_root):
    root = str(Path(bida_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from utils.dataset import HSIDataset, load_mat_hsi, sample_gt
    return HSIDataset, load_mat_hsi, sample_gt


def source_loaders(dataset_dir=DEFAULT_DATA_DIR, bida_root=DEFAULT_BIDA_ROOT, seed=2100,
                   batch_size=128, num_workers=4, patch_size=13):
    HSIDataset, load_mat_hsi, sample_gt = _bida_dataset(bida_root)
    image, gt, labels = load_mat_hsi("Houston13", str(Path(dataset_dir)) + "/", norm="normband")
    train_gt, val_gt = sample_gt(gt, 0.95, seed, mode="random")
    radius = patch_size // 2
    image = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    train_gt = np.pad(train_gt, ((radius, radius), (radius, radius)), mode="reflect")
    val_gt = np.pad(val_gt, ((radius, radius), (radius, radius)), mode="reflect")
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, drop_last=False, generator=generator)
    train = torch.utils.data.DataLoader(HSIDataset(image, train_gt, patch_size, data_aug=True), shuffle=True, **common)
    val = torch.utils.data.DataLoader(HSIDataset(image, val_gt, patch_size, data_aug=False), shuffle=False, **common)
    return train, val, len(labels), image.shape[-1]


def target_loader(dataset_dir=DEFAULT_DATA_DIR, bida_root=DEFAULT_BIDA_ROOT, seed=2100,
                  batch_size=128, num_workers=4, patch_size=13):
    HSIDataset, load_mat_hsi, sample_gt = _bida_dataset(bida_root)
    image, gt, labels = load_mat_hsi("Houston18", str(Path(dataset_dir)) + "/", norm="normband")
    target_gt, _ = sample_gt(gt, 1, seed, mode="random")
    radius = patch_size // 2
    image = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    target_gt = np.pad(target_gt, ((radius, radius), (radius, radius)), mode="reflect")
    dataset = HSIDataset(image, target_gt, patch_size, data_aug=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, drop_last=False,
                                         generator=torch.Generator().manual_seed(seed))
    return loader, len(labels)


def target_train_loader(dataset_dir=DEFAULT_DATA_DIR, bida_root=DEFAULT_BIDA_ROOT, seed=2100,
                        batch_size=128, num_workers=4, patch_size=13, generator=None):
    """Strict BiDA target centers, with labels discarded by the trainer.

    The released loader uses the target GT mask to choose eligible centers.
    This preserves the comparison protocol but is a transductive-mask caveat.
    """
    HSIDataset, load_mat_hsi, sample_gt = _bida_dataset(bida_root)
    image, gt, _ = load_mat_hsi("Houston18", str(Path(dataset_dir)) + "/", norm="normband")
    target_gt, _ = sample_gt(gt, 1, seed, mode="random")
    radius = patch_size // 2
    image = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    target_gt = np.pad(target_gt, ((radius, radius), (radius, radius)), mode="reflect")
    dataset = HSIDataset(image, target_gt, patch_size, data_aug=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, drop_last=False,
                                       generator=generator if generator is not None else torch.Generator().manual_seed(seed))
