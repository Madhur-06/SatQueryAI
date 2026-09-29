"""Spectral-index masks for optical imagery: NDWI/MNDWI water, NDBI built-up, NDVI vegetation.

All three share one pattern: compute a normalized difference on the full (unstretched)
bands, threshold it (fixed value or Otsu), drop tiny regions, and report a
threshold-margin confidence.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from satquery.io.preprocess.optical import normalized_difference
from satquery.io.validate import RasterLayer
from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.tools.classical._common import ANY_OPTICAL, clean_mask, margin_confidence, otsu
from satquery.types import BandRole, Capability, TaskType

MAPPING_TASKS = frozenset(
    {
        TaskType.VQA,
        TaskType.GROUNDING,
        TaskType.FUSION_EXTRACT,
        TaskType.CHANGE_VQA,
        TaskType.CHANGE_DESC,
    }
)
MARGIN_SCALE = 0.2
"""Index distance from the threshold at which a pixel counts as fully confident."""


class IndexMaskParams(ToolParams):
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    method: Literal["fixed", "otsu"] = "fixed"
    min_region_px: int = Field(default=4, ge=0, le=100_000)


def _index_mask(
    layer: RasterLayer,
    index: np.ndarray,
    params: IndexMaskParams,
    extra_exclude: np.ndarray | None,
    label: str,
    index_name: str,
) -> ToolOutput:
    valid = layer.valid & np.isfinite(index)
    thr = otsu(index[valid]) if params.method == "otsu" else params.threshold
    mask = valid & (index > thr)
    if extra_exclude is not None:
        mask &= ~extra_exclude
    mask = clean_mask(mask, params.min_region_px)
    n_valid = int(valid.sum())
    warnings = []
    if n_valid == 0:
        warnings.append("no valid pixels")
    return ToolOutput(
        masks={"mask": layer.georaster(mask)},
        rasters={index_name: layer.georaster(np.where(valid, index, np.nan).astype(np.float32))},
        values={
            "class": label,
            "layer": layer.key,
            "index": index_name,
            "threshold": round(float(thr), 4),
            "pixels": int(mask.sum()),
            "fraction": float(mask.sum() / n_valid) if n_valid else 0.0,
            "index_mean": float(np.nanmean(index[valid])) if n_valid else None,
        },
        confidence=margin_confidence(index, valid, thr, MARGIN_SCALE),
        confidence_basis=f"mean {index_name} distance from threshold (full at {MARGIN_SCALE})",
        warnings=warnings,
    )


class NDWIParams(IndexMaskParams):
    variant: Literal["ndwi", "mndwi", "auto"] = "auto"
    """``mndwi`` uses SWIR1 instead of NIR (better over built-up); ``auto`` picks it if present."""


class NDWIWaterMask(Tool):
    name = "ndwi_water_mask"
    version = "1.0.0"
    description = "Open-water mask from NDWI (G-NIR)/(G+NIR) or MNDWI (G-SWIR1)/(G+SWIR1)."
    tasks = MAPPING_TASKS
    input_configs = ANY_OPTICAL
    produces = frozenset({Capability.MASK_WATER})
    requires = ToolRequirements(
        n_layers=1, modality="optical", bands=((BandRole.GREEN,), (BandRole.NIR, BandRole.SWIR1))
    )
    Params = NDWIParams
    priority = 10

    def run(self, inp: ToolInput, params: NDWIParams) -> ToolOutput:
        layer = inp.layer()
        use_mndwi = params.variant != "ndwi" and layer.has(BandRole.SWIR1)
        other = BandRole.SWIR1 if use_mndwi else BandRole.NIR
        index = normalized_difference(layer.band(BandRole.GREEN), layer.band(other), layer.valid)
        out = _index_mask(layer, index, params, None, "water", "mndwi" if use_mndwi else "ndwi")
        if params.variant == "mndwi" and not use_mndwi:
            out.warnings.append("MNDWI requested but no SWIR1 band: fell back to NDWI.")
        return out


class NDBIParams(IndexMaskParams):
    max_ndvi: float | None = Field(default=0.3, ge=-1.0, le=1.0)
    """Exclude pixels greener than this (NDBI confuses some vegetation); None disables."""
    exclude_water: bool = True
    """Exclude pixels with NDWI > 0 when a green band is available."""


class NDBIBuiltupMask(Tool):
    name = "ndbi_builtup_mask"
    version = "1.0.0"
    description = (
        "Built-up mask from NDBI (SWIR1-NIR)/(SWIR1+NIR), excluding vegetated (NDVI) and "
        "water (NDWI) pixels. Needs a SWIR band."
    )
    tasks = MAPPING_TASKS
    input_configs = ANY_OPTICAL
    produces = frozenset({Capability.MASK_BUILTUP})
    requires = ToolRequirements(
        n_layers=1, modality="optical", bands=((BandRole.SWIR1,), (BandRole.NIR,))
    )
    Params = NDBIParams
    priority = 10

    def run(self, inp: ToolInput, params: NDBIParams) -> ToolOutput:
        layer = inp.layer()
        nir = layer.band(BandRole.NIR)
        index = normalized_difference(layer.band(BandRole.SWIR1), nir, layer.valid)
        exclude = np.zeros(layer.shape, dtype=bool)
        if params.max_ndvi is not None and layer.has(BandRole.RED):
            ndvi = normalized_difference(nir, layer.band(BandRole.RED), layer.valid)
            exclude |= np.nan_to_num(ndvi, nan=-1) > params.max_ndvi
        if params.exclude_water and layer.has(BandRole.GREEN):
            ndwi = normalized_difference(layer.band(BandRole.GREEN), nir, layer.valid)
            exclude |= np.nan_to_num(ndwi, nan=-1) > 0
        return _index_mask(layer, index, params, exclude, "builtup", "ndbi")


class NDVIParams(IndexMaskParams):
    threshold: float = Field(default=0.4, ge=-1.0, le=1.0)


class NDVIVegetationMask(Tool):
    name = "ndvi_vegetation_mask"
    version = "1.0.0"
    description = "Vegetation mask from NDVI (NIR-R)/(NIR+R)."
    tasks = MAPPING_TASKS
    input_configs = ANY_OPTICAL
    produces = frozenset({Capability.MASK_VEGETATION})
    requires = ToolRequirements(
        n_layers=1, modality="optical", bands=((BandRole.NIR,), (BandRole.RED,))
    )
    Params = NDVIParams
    priority = 10

    def run(self, inp: ToolInput, params: NDVIParams) -> ToolOutput:
        layer = inp.layer()
        index = normalized_difference(
            layer.band(BandRole.NIR), layer.band(BandRole.RED), layer.valid
        )
        return _index_mask(layer, index, params, None, "vegetation", "ndvi")
