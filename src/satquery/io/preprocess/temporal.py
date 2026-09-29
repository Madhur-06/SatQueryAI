"""Bi-temporal preprocessing: radiometric normalisation of date 2 to date 1."""

from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms


def histogram_match(
    src: np.ndarray, ref: np.ndarray, src_valid: np.ndarray, ref_valid: np.ndarray
) -> np.ndarray:
    """Match each band of ``src`` (date 2) to ``ref`` (date 1), using valid pixels only.

    Caveat: rank-based matching assumes both dates have the same class proportions. A
    change covering even a few percent of the scene shifts every other class (on the
    synthetic pair, 7 % new built-up made unchanged water look changed). Prefer
    ``linear_normalise`` for change detection; this stays available as an option.
    """
    if src.shape[0] != ref.shape[0]:
        raise ValueError("src and ref must have the same number of bands")
    out = src.astype(np.float32).copy()
    both = src_valid & ref_valid
    if not both.any():
        return out
    for b in range(src.shape[0]):
        out[b][src_valid] = match_histograms(src[b][src_valid], ref[b][ref_valid])
    return out


def linear_normalise(
    src: np.ndarray,
    ref: np.ndarray,
    src_valid: np.ndarray,
    ref_valid: np.ndarray,
    iterations: int = 5,
    k: float = 3.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Relative radiometric normalisation robust to real change.

    Per band, fit ``src = gain * ref + offset`` on co-valid pixels, then refit on pixels
    whose residual is within ``k`` robust standard deviations (MAD), so changed pixels
    stop influencing the fit. Returns ``src`` mapped onto ``ref``'s scale and the
    per-band ``(gain, offset)``.
    """
    both = src_valid & ref_valid
    out = src.astype(np.float32).copy()
    coeffs: list[tuple[float, float]] = []
    for b in range(src.shape[0]):
        x = ref[b][both].astype(np.float64)
        y = src[b][both].astype(np.float64)
        keep = np.ones(x.shape, dtype=bool)
        gain, offset = 1.0, 0.0
        for _ in range(iterations):
            if keep.sum() < 2 or np.ptp(x[keep]) == 0:
                break
            gain, offset = np.polyfit(x[keep], y[keep], 1)
            resid = y - (gain * x + offset)
            mad = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
            new_keep = np.abs(resid) <= k * max(mad, 1e-9)
            if (new_keep == keep).all():
                break
            keep = new_keep
        if abs(gain) < 1e-6:
            gain, offset = 1.0, 0.0
        out[b] = np.where(src_valid, (src[b] - offset) / gain, src[b])
        coeffs.append((float(gain), float(offset)))
    return out, coeffs
