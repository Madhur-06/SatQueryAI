"""Bi-temporal preprocessing: radiometric normalisation of date 2 to date 1."""

from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms


def histogram_match(
    src: np.ndarray, ref: np.ndarray, src_valid: np.ndarray, ref_valid: np.ndarray
) -> np.ndarray:
    """Match each band of ``src`` (date 2) to ``ref`` (date 1), using valid pixels only.

    Caveat: a change that covers a large share of the scene also shifts the histogram
    and is partly normalised away. The planner therefore uses this only as CVA input,
    never for index masks, and records it as a tool parameter.
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
