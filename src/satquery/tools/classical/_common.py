"""Helpers shared by the classical tools."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

from satquery.types import InputConfig

ANY_OPTICAL = frozenset(
    {InputConfig.SINGLE_OPT, InputConfig.PAIR_OPT_SAR, InputConfig.PAIR_TEMPORAL}
)
ANY_SAR = frozenset({InputConfig.SINGLE_SAR, InputConfig.PAIR_OPT_SAR, InputConfig.PAIR_TEMPORAL})


def clean_mask(mask: np.ndarray, min_region_px: int) -> np.ndarray:
    """Drop connected regions smaller than ``min_region_px`` pixels (8-connectivity)."""
    if min_region_px <= 1 or not mask.any():
        return mask
    return remove_small_objects(mask, max_size=min_region_px - 1, connectivity=2)


def otsu(values: np.ndarray) -> float:
    """Otsu threshold of a 1-D sample (falls back to the median for constant data)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if np.ptp(finite) == 0:
        return float(finite[0])
    return float(threshold_otsu(finite))


def margin_confidence(
    index: np.ndarray, valid: np.ndarray, threshold: float, scale: float
) -> float:
    """Mean of ``min(1, |index - threshold| / scale)`` over valid pixels.

    A threshold-based classifier is trustworthy when few pixels sit near its threshold:
    1.0 means every pixel is at least ``scale`` away from it, 0.0 that all sit on it.
    """
    values = index[valid & np.isfinite(index)]
    if values.size == 0:
        return 0.0
    return float(np.mean(np.minimum(1.0, np.abs(values - threshold) / scale)))


def bimodality(values: np.ndarray, threshold: float) -> float:
    """Otsu's between-class variance ratio (eta) in [0, 1]: 1 = two perfectly separated modes."""
    finite = values[np.isfinite(values)]
    total = float(np.var(finite)) if finite.size else 0.0
    if total == 0:
        return 0.0
    lo, hi = finite[finite <= threshold], finite[finite > threshold]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w0, w1 = lo.size / finite.size, hi.size / finite.size
    between = w0 * w1 * (float(lo.mean()) - float(hi.mean())) ** 2
    return float(np.clip(between / total, 0.0, 1.0))
