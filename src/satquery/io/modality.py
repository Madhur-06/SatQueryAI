"""Modality detection from metadata and pixel statistics.

Evidence is ranked: band descriptions (strongest) > sensor tags > pixel statistics.
Nothing here assumes a particular sensor's band order; callers can always override.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import uniform_filter

from satquery.io.raster import RasterMeta
from satquery.types import Modality

_SAR_DESC = re.compile(
    r"(?<![a-z])(vv|vh|hh|hv|sigma0?|gamma0?|beta0|backscatter|sar)(?![a-z])", re.IGNORECASE
)
_OPT_DESC = re.compile(
    r"(?<![a-z])(blue|green|red|nir|near.?infra.?red|swir|red.?edge|pan|panchromatic|coastal)"
    r"(?![a-z])",
    re.IGNORECASE,
)
_SAR_SENSOR = re.compile(
    r"sentinel.?1|(?<![a-z0-9])s1[ab]?(?![a-z0-9])|radarsat|risat|eos.?04|palsar|terrasar"
    r"|cosmo.?skymed|iceye|capella|nisar|(?<![a-z])sar(?![a-z])",
    re.IGNORECASE,
)
_OPT_SENSOR = re.compile(
    r"sentinel.?2|(?<![a-z0-9])s2[ab]?(?![a-z0-9])|landsat|resourcesat|cartosat|liss|awifs"
    r"|planetscope|spot.?[67]|worldview|pleiades|modis",
    re.IGNORECASE,
)
_SENSOR_KEYS = ("sensor", "mission", "platform", "spacecraft", "satellite", "instrument", "product")

SPECKLE_CV_THRESHOLD = 0.3
"""Median local coefficient of variation above which texture looks like SAR speckle."""


class ModalityDecision(BaseModel):
    modality: Modality
    confidence: float
    reasons: list[str]


def _sample(data: np.ndarray, valid: np.ndarray, max_pixels: int = 250_000) -> np.ndarray:
    """(bands, n) sample of valid pixels."""
    values = data[:, valid]
    if values.shape[1] > max_pixels:
        idx = np.random.default_rng(0).choice(values.shape[1], max_pixels, replace=False)
        values = values[:, idx]
    return values


def local_cv(band: np.ndarray, valid: np.ndarray, size: int = 5) -> float:
    """Median coefficient of variation in ``size`` x ``size`` windows (speckle indicator)."""
    x = np.where(valid, band, 0.0).astype(np.float64)
    if np.nanmin(x[valid]) < 0:  # dB: evaluate texture on linear power
        x = np.where(valid, 10 ** (x / 10), 0.0)
    mean = uniform_filter(x, size)
    sq = uniform_filter(x * x, size)
    std = np.sqrt(np.clip(sq - mean**2, 0, None))
    interior = valid & (mean > 0)
    if not interior.any():
        return 0.0
    return float(np.median(std[interior] / mean[interior]))


def _family(meta: RasterMeta) -> tuple[Modality, float, str] | None:
    """Evidence from descriptions and tags only."""
    descs = [d for d in meta.band_descriptions if d] + [
        v for t in meta.band_tags for k, v in t.items() if "polari" in k.lower()
    ]
    sar_hits = sum(bool(_SAR_DESC.search(d)) for d in descs)
    opt_hits = sum(bool(_OPT_DESC.search(d)) for d in descs)
    if sar_hits > opt_hits:
        return Modality.SAR, 0.95, f"band descriptions look like SAR polarisations {descs}"
    if opt_hits > sar_hits:
        return Modality.OPTICAL, 0.95, f"band descriptions name optical bands {descs}"

    sensor_text = " ".join(
        v for k, v in meta.tags.items() if any(s in k.lower() for s in _SENSOR_KEYS)
    )
    if sensor_text:
        if _SAR_SENSOR.search(sensor_text):
            return Modality.SAR, 0.9, f"sensor tags indicate SAR ({sensor_text!r})"
        if _OPT_SENSOR.search(sensor_text):
            return Modality.OPTICAL, 0.9, f"sensor tags indicate optical ({sensor_text!r})"
    return None


def _from_statistics(
    meta: RasterMeta, data: np.ndarray, valid: np.ndarray
) -> tuple[Modality, float, str]:
    if not valid.any():
        return Modality.UNKNOWN, 0.0, "no valid pixels"
    sample = _sample(data, valid)
    is_float = meta.dtype.startswith("float")
    median = float(np.median(sample))
    p1, p99 = (float(v) for v in np.percentile(sample, [1, 99]))
    cv = local_cv(data[0], valid)

    if is_float and meta.count <= 4 and median < 0 and p1 > -60 and p99 < 25:
        return Modality.SAR, 0.8, f"float values in a dB range (median {median:.1f}, p1 {p1:.1f})"
    if meta.count <= 2 and cv > SPECKLE_CV_THRESHOLD:
        reason = f"speckle-like texture (local CV {cv:.2f}) on {meta.count} band(s)"
        return Modality.SAR, 0.65, reason
    if meta.count >= 3:
        kind = "reflectance" if is_float else "digital numbers"
        return Modality.OPTICAL, 0.6, f"{meta.count} smooth bands of {kind} (local CV {cv:.2f})"
    if meta.count == 1 and not is_float:
        return Modality.OPTICAL, 0.4, "single integer band: assumed panchromatic optical"
    return Modality.UNKNOWN, 0.0, f"{meta.count} band(s) of {meta.dtype}: no decisive evidence"


def detect_modality(meta: RasterMeta, data: np.ndarray, valid: np.ndarray) -> ModalityDecision:
    """Classify an image as OPTICAL, MULTISPECTRAL, SAR or UNKNOWN."""
    found = _family(meta)
    modality, confidence, reason = found if found else _from_statistics(meta, data, valid)
    reasons = [reason]
    if modality is Modality.OPTICAL and meta.count >= 5:
        modality = Modality.MULTISPECTRAL
        reasons.append(f"{meta.count} bands => multispectral")
    return ModalityDecision(modality=modality, confidence=confidence, reasons=reasons)
