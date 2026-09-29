"""SAR open-water mask: Lee-filtered co-pol backscatter (dB), Otsu threshold."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, field_validator

from satquery.io.preprocess.sar import lee_filter, to_db, to_linear
from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.tools.classical._common import (
    ANY_SAR,
    bimodality,
    clean_mask,
    margin_confidence,
    otsu,
)
from satquery.tools.classical.indices import MAPPING_TASKS
from satquery.types import BandRole, Capability

# Open water is specular: typically below about -15 dB in VV. An Otsu threshold far above
# that means the histogram has no dark water mode, and the "water" is just darker land.
MAX_PLAUSIBLE_THRESHOLD_DB = -12.0
MARGIN_SCALE_DB = 3.0


class SarWaterParams(ToolParams):
    polarisation: Literal["auto", "vv", "hh"] = "auto"
    speckle_filter: Literal["lee", "none"] = "lee"
    window: int = Field(default=5, ge=3, le=15)
    threshold_db: float | None = Field(default=None, ge=-40.0, le=0.0)
    """Fixed threshold; None = Otsu on the clipped histogram."""
    clip_db: tuple[float, float] = (-35.0, 5.0)
    min_region_px: int = Field(default=9, ge=0, le=100_000)

    @field_validator("window")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("window must be odd")
        return v

    @field_validator("clip_db")
    @classmethod
    def _ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not -60.0 <= v[0] < v[1] <= 30.0:
            raise ValueError("clip_db must be (low, high) with -60 <= low < high <= 30")
        return v


class SarWaterOtsu(Tool):
    name = "sar_water_otsu"
    version = "1.0.0"
    description = (
        "Open-water mask from co-pol (VV or HH) backscatter: Lee speckle filter, "
        "Otsu threshold in dB, plausibility check."
    )
    tasks = MAPPING_TASKS
    input_configs = ANY_SAR
    produces = frozenset({Capability.MASK_WATER})
    requires = ToolRequirements(n_layers=1, modality="sar", bands=((BandRole.VV, BandRole.HH),))
    Params = SarWaterParams
    priority = 10

    def run(self, inp: ToolInput, params: SarWaterParams) -> ToolOutput:
        layer = inp.layer()
        if params.polarisation == "auto":
            role = BandRole.VV if layer.has(BandRole.VV) else BandRole.HH
        else:
            role = BandRole(params.polarisation)
        db = layer.band(role)
        valid = layer.valid
        if params.speckle_filter == "lee":
            db = to_db(lee_filter(to_linear(db), valid, size=params.window))
        lo, hi = params.clip_db
        clipped = np.clip(db, lo, hi)

        warnings: list[str] = []
        if params.threshold_db is not None:
            thr = params.threshold_db
        else:
            thr = otsu(clipped[valid])
            if thr > MAX_PLAUSIBLE_THRESHOLD_DB:
                warnings.append(
                    f"Otsu threshold {thr:.1f} dB is implausibly high for open water; "
                    f"capped at {MAX_PLAUSIBLE_THRESHOLD_DB} dB (scene may contain no water)."
                )
                thr = MAX_PLAUSIBLE_THRESHOLD_DB
        mask = clean_mask(valid & (clipped < thr), params.min_region_px)
        n_valid = int(valid.sum())

        separation = bimodality(clipped[valid], thr)
        margin = margin_confidence(clipped, valid, thr, MARGIN_SCALE_DB)
        confidence = margin * (0.5 + 0.5 * separation)
        if warnings:
            confidence *= 0.5
        return ToolOutput(
            masks={"mask": layer.georaster(mask)},
            rasters={"backscatter_db": layer.georaster(np.where(valid, db, np.nan))},
            values={
                "class": "water",
                "layer": layer.key,
                "polarisation": str(role),
                "threshold_db": round(float(thr), 2),
                "bimodality": round(separation, 3),
                "pixels": int(mask.sum()),
                "fraction": float(mask.sum() / n_valid) if n_valid else 0.0,
            },
            confidence=float(confidence),
            confidence_basis=(
                f"threshold margin (full at {MARGIN_SCALE_DB} dB) x histogram bimodality"
            ),
            warnings=warnings,
        )
