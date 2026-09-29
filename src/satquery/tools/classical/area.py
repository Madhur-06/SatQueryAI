"""Area statistics for every mask of an upstream output: pixels, hectares, share, regions."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from scipy.ndimage import label as label_regions

from satquery.io.geometry import pixel_area_m2
from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.types import Capability, InputConfig, TaskType


class AreaStatsParams(ToolParams):
    masks: list[str] | None = None
    """Which upstream mask keys to measure; None = all of them."""
    min_region_px: int = Field(default=1, ge=1, le=100_000)
    """Regions smaller than this are not counted as separate objects."""
    connectivity: Literal[4, 8] = 8


class AreaStats(Tool):
    name = "area_stats"
    version = "1.0.0"
    description = (
        "Measures masks: pixel count, area in hectares (from the geotransform; geodesic "
        "for geographic CRSs), share of the valid scene, and number of connected regions."
    )
    tasks = frozenset(TaskType)
    input_configs = frozenset(InputConfig)
    produces = frozenset({Capability.AREA_STATS})
    requires = ToolRequirements(upstream={"source": None})
    Params = AreaStatsParams
    priority = 10

    def run(self, inp: ToolInput, params: AreaStatsParams) -> ToolOutput:
        source = inp.upstream["source"]
        keys = params.masks if params.masks is not None else list(source.masks)
        missing = [k for k in keys if k not in source.masks]
        if missing:
            raise ValueError(f"upstream has no masks {missing}; available: {list(source.masks)}")
        structure = np.ones((3, 3)) if params.connectivity == 8 else None
        valid = inp.bundle.reference.valid
        n_valid = int(valid.sum())
        values: dict[str, float | int | str | bool | None] = {}
        warnings: list[str] = []
        for key in keys:
            grid = source.masks[key]
            mask = grid.array.astype(bool)
            px_area = pixel_area_m2(grid.affine, grid.crs, mask.shape[0])
            labels, n = label_regions(mask, structure=structure)
            if n and params.min_region_px > 1:
                sizes = np.bincount(labels.ravel())[1:]
                n = int((sizes >= params.min_region_px).sum())
            values[f"{key}_pixels"] = int(mask.sum())
            values[f"{key}_fraction"] = float(mask.sum() / n_valid) if n_valid else 0.0
            values[f"{key}_regions"] = int(n)
            if px_area is None:
                values[f"{key}_hectares"] = None
            else:
                values[f"{key}_hectares"] = float((mask * px_area).sum() / 10_000)
        if any(source.masks[k].crs is None for k in keys):
            warnings.append("No CRS: areas are reported in pixels only.")
        values["class"] = source.values.get("class")
        values["pixel_size_m"] = _pixel_size(source.masks[keys[0]]) if keys else None
        return ToolOutput(
            values=values,
            confidence=source.confidence,
            confidence_basis="inherited from the measured masks (measurement itself is exact)",
            warnings=warnings,
        )


def _pixel_size(grid) -> float | None:
    area = pixel_area_m2(grid.affine, grid.crs, 1)
    return None if area is None else round(float(np.sqrt(area[0, 0])), 3)
