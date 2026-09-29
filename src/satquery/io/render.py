"""Quicklooks and mask overlays as PNG (for the UI, reports and VLM input)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from satquery.io.preprocess import optical, sar
from satquery.types import BandRole, Modality

# RGB colours per mask label, first match wins (specific keys before class names).
MASK_COLOURS: dict[str, tuple[int, int, int]] = {
    "optical_only": (255, 159, 28),
    "sar_only": (155, 93, 229),
    "gain": (230, 57, 70),
    "loss": (255, 140, 0),
    "change": (255, 214, 10),
    "water": (30, 144, 255),
    "built": (230, 57, 70),
    "vegetation": (46, 204, 113),
}
DEFAULT_COLOUR = (255, 0, 255)


def colour_for(label: str) -> tuple[int, int, int]:
    for key, colour in MASK_COLOURS.items():
        if key in label.lower():
            return colour
    return DEFAULT_COLOUR


def quicklook(
    data: np.ndarray, valid: np.ndarray, modality: Modality, band_map: dict[BandRole, int]
) -> np.ndarray:
    """uint8 ``(H, W, 3)`` display image: true colour, SAR composite, or a fallback."""
    if modality is Modality.SAR:
        co = band_map.get(BandRole.VV, band_map.get(BandRole.HH, 0))
        cross = band_map.get(BandRole.VH, band_map.get(BandRole.HV))
        if cross is None and data.shape[0] == 2 and not band_map:
            cross = 1
        return sar.composite(data[co], None if cross is None else data[cross], valid)
    rgb = optical.true_colour(data, band_map, valid)
    return rgb if rgb is not None else optical.fallback_rgb(data, valid)


def mask_rgba(mask: np.ndarray, colour: tuple[int, int, int], alpha: int = 255) -> np.ndarray:
    """Transparent RGBA image with the mask painted in ``colour`` (for UI layering)."""
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[mask.astype(bool)] = (*colour, alpha)
    return out


def blend(
    rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float = 0.5
) -> np.ndarray:
    """``rgb`` with the mask alpha-blended on top (for static reports)."""
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(colour, dtype=np.float32)
    return out.round().astype(np.uint8)


def save_png(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, optimize=True)
    return path
