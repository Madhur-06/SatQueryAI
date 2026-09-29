"""Optical preprocessing: percentile stretch and RGB / false-colour renderings.

Renderings are for display and VLM input only; index tools always use the full,
unstretched bands.
"""

from __future__ import annotations

import numpy as np

from satquery.types import BandRole


def percentile_stretch(
    band: np.ndarray, valid: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Linear stretch between the given percentiles of valid pixels, to ``[0, 1]``."""
    values = band[valid]
    if values.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~valid] = 0.0
    return out


def render(
    data: np.ndarray,
    band_map: dict[BandRole, int],
    valid: np.ndarray,
    roles: tuple[BandRole, BandRole, BandRole],
) -> np.ndarray | None:
    """uint8 ``(H, W, 3)`` image from three band roles, or None if any role is missing."""
    if not all(r in band_map for r in roles):
        return None
    channels = [percentile_stretch(data[band_map[r]], valid) for r in roles]
    return (np.stack(channels, axis=-1) * 255).round().astype(np.uint8)


def true_colour(data: np.ndarray, band_map: dict[BandRole, int], valid: np.ndarray):
    return render(data, band_map, valid, (BandRole.RED, BandRole.GREEN, BandRole.BLUE))


def false_colour(data: np.ndarray, band_map: dict[BandRole, int], valid: np.ndarray):
    """NIR-R-G: vegetation appears red, water dark."""
    return render(data, band_map, valid, (BandRole.NIR, BandRole.RED, BandRole.GREEN))


def fallback_rgb(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Greyscale (1-2 bands) or the first three bands, when band roles are unknown."""
    idx = (0, 0, 0) if data.shape[0] < 3 else (0, 1, 2)
    channels = [percentile_stretch(data[i], valid) for i in idx]
    return (np.stack(channels, axis=-1) * 255).round().astype(np.uint8)


def normalized_difference(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)`` with invalid / zero-sum pixels set to NaN."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    ok = valid & (denom != 0)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out
