"""SAR preprocessing: scale detection, dB conversion, Lee speckle filter, composites."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import uniform_filter

SarScale = Literal["db", "linear", "dn"]

DB_FLOOR = -50.0
# Display ranges (dB) for the VV / VH / VV-VH composite.
VV_RANGE = (-25.0, 0.0)
VH_RANGE = (-30.0, -5.0)
RATIO_RANGE = (0.0, 15.0)


def detect_scale(data: np.ndarray, valid: np.ndarray, dtype: str) -> SarScale:
    """dB if any valid value is negative; integer data is treated as amplitude DN."""
    if not dtype.startswith("float"):
        return "dn"
    values = data[:, valid] if data.ndim == 3 else data[valid]
    return "db" if values.size and float(values.min()) < 0 else "linear"


def to_db(linear: np.ndarray) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.maximum(linear, 1e-10)), DB_FLOOR).astype(np.float32)


def to_linear(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 10.0).astype(np.float32)


def to_linear_power(data: np.ndarray, scale: SarScale) -> np.ndarray:
    """Convert any supported scale to linear power (DN is squared: uncalibrated)."""
    if scale == "db":
        return to_linear(data)
    if scale == "dn":
        return np.square(data.astype(np.float32))
    return data.astype(np.float32)


def estimate_looks(linear: np.ndarray, valid: np.ndarray, size: int = 7) -> float:
    """Equivalent number of looks: ``1 / median(local CV²)`` over ``size`` x ``size`` windows.

    The median is robust to edges between classes (tested within ~5 % on synthetic
    gamma speckle, homogeneous or two-class). Lower percentiles bias L upwards.
    """
    mean = uniform_filter(np.where(valid, linear, 0.0), size)
    sq = uniform_filter(np.where(valid, linear**2, 0.0), size)
    var = np.clip(sq - mean**2, 0, None)
    full = uniform_filter(valid.astype(np.float64), size) > 0.999  # windows fully valid
    ok = full & (mean > 0)
    if not ok.any():
        return 1.0
    cv2 = var[ok] / mean[ok] ** 2
    return float(np.clip(1.0 / max(float(np.median(cv2)), 1e-6), 1.0, 50.0))


def lee_filter(
    linear: np.ndarray, valid: np.ndarray, size: int = 5, looks: float | None = None
) -> np.ndarray:
    """Classic Lee filter on linear intensity.

    ``out = m + w (x - m)`` with ``w = clip(1 - Cu² / Ci², 0, 1)``, where ``Cu² = 1/L``
    is the speckle variation and ``Ci`` the local coefficient of variation. ``looks``
    (L) is estimated from homogeneous areas when not given.
    TODO(phase 2): refined Lee (edge-aligned windows) for sharper field boundaries.
    """
    if size % 2 == 0 or size < 3:
        raise ValueError("size must be an odd integer >= 3")
    x = np.where(valid, linear, 0.0).astype(np.float64)
    weight = uniform_filter(valid.astype(np.float64), size)
    safe = np.maximum(weight, 1e-12)
    mean = uniform_filter(x, size) / safe
    sq = uniform_filter(x * x, size) / safe
    var = np.clip(sq - mean**2, 0, None)
    L = looks if looks is not None else estimate_looks(linear, valid)
    cu2 = 1.0 / L
    ci2 = np.where(mean > 0, var / np.maximum(mean**2, 1e-20), 0.0)
    w = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-12), 0.0, 1.0)
    out = mean + w * (x - mean)
    return np.where(valid, out, 0.0).astype(np.float32)


def normalise(db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to ``[lo, hi]`` dB and scale to ``[0, 1]``."""
    return ((np.clip(db, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def composite(vv_db: np.ndarray, vh_db: np.ndarray | None, valid: np.ndarray) -> np.ndarray:
    """uint8 RGB: R = VV, G = VH, B = VV/VH ratio (VV - VH in dB). Grey if VH is missing."""
    r = normalise(vv_db, *VV_RANGE)
    if vh_db is None:
        rgb = np.stack([r, r, r], axis=-1)
    else:
        g = normalise(vh_db, *VH_RANGE)
        b = normalise(vv_db - vh_db, *RATIO_RANGE)
        rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    return (rgb * 255).round().astype(np.uint8)
