"""Qwen3.5 image patchify in numpy (no torchvision in the data workers).

Reproduces ``Qwen2VLImageProcessorFast`` for an image that is already at the target size
(a multiple of ``patch * merge``): rescale 1/255, normalise with mean/std 0.5, split into
``patch x patch`` tiles grouped by ``merge x merge`` blocks, and duplicate along the
temporal axis (``temporal_patch_size`` 2, a still image).

Workers ship ``uint8`` tiles ``[n_patches, 3, 16, 16]`` (8x smaller than the float32
processor output). ``to_pixel_values`` finishes normalisation and temporal duplication on
the training device. ``tests/benvqa/test_vision.py`` checks bit-exactness against the
official processor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PATCH = 16
MERGE = 2
TEMPORAL = 2
MEAN = 0.5
STD = 0.5


def grid_for(size: int) -> tuple[int, int, int]:
    if size % (PATCH * MERGE):
        raise ValueError(f"image size {size} must be a multiple of {PATCH * MERGE}")
    g = size // PATCH
    return (1, g, g)


def tokens_per_image(size: int) -> int:
    _, gh, gw = grid_for(size)
    return gh * gw // (MERGE * MERGE)


def patchify_u8(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """``[H, W, 3]`` uint8 -> (``[gh*gw, 3, 16, 16]`` uint8 in Qwen order, grid_thw)."""
    h, w, c = img.shape
    if h != w:
        raise ValueError("square images only")
    _, gh, gw = grid_for(h)
    x = np.ascontiguousarray(img.transpose(2, 0, 1))  # C, H, W
    x = x.reshape(c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)  # gh/m, gw/m, m, m, C, p, p
    return np.ascontiguousarray(x).reshape(gh * gw, c, PATCH, PATCH), (1, gh, gw)


def to_pixel_values(tiles_u8: Any, dtype: Any = None) -> Any:
    """uint8 tiles (numpy or torch) -> float ``[n, 3 * TEMPORAL * 16 * 16]`` model input."""
    try:
        import torch
    except ImportError:  # pragma: no cover - numpy path for CPU-only checks
        torch = None
    if torch is not None and isinstance(tiles_u8, torch.Tensor):
        x = tiles_u8.to(torch.float32) * (1 / 255.0)
        x = (x - MEAN) / STD
        n, c, p, q = x.shape
        x = x.unsqueeze(2).expand(n, c, TEMPORAL, p, q).reshape(n, c * TEMPORAL * p * q)
        return x.to(dtype) if dtype is not None else x
    x = np.asarray(tiles_u8, dtype=np.float32) * np.float32(1 / 255.0)
    x = (x - np.float32(MEAN)) / np.float32(STD)
    n, c, p, q = x.shape
    x = np.broadcast_to(x[:, :, None], (n, c, TEMPORAL, p, q)).reshape(n, c * TEMPORAL * p * q)
    return x.astype(dtype) if dtype is not None else x
