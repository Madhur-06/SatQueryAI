"""Bi-temporal change detection: CVA (optical), log-ratio (SAR), class-mask comparison."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from satquery.io.preprocess.sar import lee_filter, to_db, to_linear
from satquery.io.preprocess.temporal import histogram_match, linear_normalise
from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.tools.classical._common import bimodality, clean_mask, margin_confidence, otsu
from satquery.types import BandRole, Capability, InputConfig, TaskType

CHANGE_TASKS = frozenset({TaskType.CHANGE_VQA, TaskType.CHANGE_DESC})
TEMPORAL = frozenset({InputConfig.PAIR_TEMPORAL})


class CVAParams(ToolParams):
    normalisation: Literal["linear", "histogram", "none"] = "linear"
    """Radiometric normalisation of date 2 to date 1 before differencing. ``linear`` is a
    robust gain/offset fit that ignores changed pixels; ``histogram`` (rank matching)
    can turn real change elsewhere into false change and is kept for comparison."""
    standardise: bool = True
    """Scale each band by its date-1 standard deviation so no band dominates."""
    threshold: float | None = Field(default=None, ge=0.0)
    """Fixed magnitude threshold; None = Otsu."""
    min_region_px: int = Field(default=9, ge=0, le=100_000)


class ChangeCVA(Tool):
    name = "change_cva"
    version = "1.0.0"
    description = (
        "Change vector analysis on optical pairs: magnitude of the per-pixel spectral "
        "difference over the common bands, Otsu-thresholded into a change mask."
    )
    tasks = CHANGE_TASKS
    input_configs = TEMPORAL
    produces = frozenset({Capability.MASK_CHANGE})
    requires = ToolRequirements(n_layers=2, modality="optical")
    Params = CVAParams
    priority = 10

    def run(self, inp: ToolInput, params: CVAParams) -> ToolOutput:
        t1, t2 = inp.layer(0), inp.layer(1)
        common = [r for r in t1.band_map if r in t2.band_map]
        if common:
            idx1 = [t1.band_map[r] for r in common]
            idx2 = [t2.band_map[r] for r in common]
            bands_used = [str(r) for r in common]
        else:  # validation guarantees equal band counts in this case
            idx1 = idx2 = list(range(t1.data.shape[0]))
            bands_used = [f"band{i + 1}" for i in idx1]
        a, b = t1.data[idx1], t2.data[idx2]
        valid = t1.valid & t2.valid
        gains: list[float] = []
        if params.normalisation == "linear":
            b, coeffs = linear_normalise(b, a, t2.valid, t1.valid)
            gains = [g for g, _ in coeffs]
        elif params.normalisation == "histogram":
            b = histogram_match(b, a, t2.valid, t1.valid)
        diff = b - a
        if params.standardise:
            std = np.array([a[i][valid].std() if valid.any() else 1.0 for i in range(len(a))])
            diff = diff / np.maximum(std, 1e-6)[:, None, None]
        magnitude = np.sqrt(np.sum(diff**2, axis=0))
        thr = params.threshold if params.threshold is not None else otsu(magnitude[valid])
        mask = clean_mask(valid & (magnitude > thr), params.min_region_px)
        n_valid = int(valid.sum())
        sep = bimodality(magnitude[valid], thr)
        return ToolOutput(
            masks={"mask": t1.georaster(mask)},
            rasters={"magnitude": t1.georaster(np.where(valid, magnitude, np.nan))},
            values={
                "class": "change",
                "bands": ",".join(bands_used),
                "threshold": round(float(thr), 4),
                "normalisation": params.normalisation,
                "mean_gain": round(float(np.mean(gains)), 4) if gains else None,
                "pixels": int(mask.sum()),
                "fraction": float(mask.sum() / n_valid) if n_valid else 0.0,
            },
            confidence=margin_confidence(magnitude, valid, thr, max(thr, 1e-6) * 0.5)
            * (0.5 + 0.5 * sep),
            confidence_basis="magnitude margin from threshold x histogram bimodality",
        )


class LogRatioParams(ToolParams):
    polarisation: Literal["auto", "vv", "vh", "hh", "hv"] = "auto"
    speckle_filter: Literal["lee", "none"] = "lee"
    window: int = Field(default=5, ge=3, le=15)
    threshold_db: float | None = Field(default=None, ge=0.0, le=30.0)
    """Fixed |log-ratio| threshold in dB; None = Otsu."""
    min_region_px: int = Field(default=9, ge=0, le=100_000)


class SarLogRatioChange(Tool):
    name = "sar_logratio_change"
    version = "1.0.0"
    description = (
        "SAR change detection: absolute log-ratio (dB difference) of speckle-filtered "
        "backscatter between dates, Otsu-thresholded. Reports increase / decrease shares."
    )
    tasks = CHANGE_TASKS
    input_configs = TEMPORAL
    produces = frozenset({Capability.MASK_CHANGE})
    requires = ToolRequirements(
        n_layers=2,
        modality="sar",
        bands=((BandRole.VV, BandRole.HH, BandRole.VH, BandRole.HV),),
    )
    Params = LogRatioParams
    priority = 10

    def run(self, inp: ToolInput, params: LogRatioParams) -> ToolOutput:
        t1, t2 = inp.layer(0), inp.layer(1)
        if params.polarisation == "auto":
            order = (BandRole.VV, BandRole.HH, BandRole.VH, BandRole.HV)
            role = next(r for r in order if t1.has(r) and t2.has(r))
        else:
            role = BandRole(params.polarisation)
        valid = t1.valid & t2.valid
        db1, db2 = t1.band(role), t2.band(role)
        if params.speckle_filter == "lee":
            db1 = to_db(lee_filter(to_linear(db1), valid, params.window))
            db2 = to_db(lee_filter(to_linear(db2), valid, params.window))
        ratio = db2 - db1
        magnitude = np.abs(ratio)
        thr = params.threshold_db if params.threshold_db is not None else otsu(magnitude[valid])
        mask = clean_mask(valid & (magnitude > thr), params.min_region_px)
        n_valid = int(valid.sum())
        n_mask = int(mask.sum())
        sep = bimodality(magnitude[valid], thr)
        return ToolOutput(
            masks={
                "mask": t1.georaster(mask),
                "increase": t1.georaster(mask & (ratio > 0)),
                "decrease": t1.georaster(mask & (ratio < 0)),
            },
            rasters={"log_ratio_db": t1.georaster(np.where(valid, ratio, np.nan))},
            values={
                "class": "change",
                "polarisation": str(role),
                "threshold_db": round(float(thr), 2),
                "pixels": n_mask,
                "fraction": float(n_mask / n_valid) if n_valid else 0.0,
                "increase_pixels": int((mask & (ratio > 0)).sum()),
                "decrease_pixels": int((mask & (ratio < 0)).sum()),
            },
            confidence=margin_confidence(magnitude, valid, thr, 3.0) * (0.5 + 0.5 * sep),
            confidence_basis="|log-ratio| margin (full at 3 dB) x histogram bimodality",
        )


class MaskCompareParams(ToolParams):
    min_region_px: int = Field(default=0, ge=0, le=100_000)


class MaskCompare(Tool):
    name = "mask_compare"
    version = "1.0.0"
    description = (
        "Post-classification comparison of one class between two dates: gain (absent -> "
        "present) and loss (present -> absent) masks, net change."
    )
    tasks = CHANGE_TASKS
    input_configs = TEMPORAL
    produces = frozenset({Capability.MASK_CLASS_CHANGE})
    requires = ToolRequirements(upstream={"before": None, "after": None})
    Params = MaskCompareParams
    priority = 10

    def run(self, inp: ToolInput, params: MaskCompareParams) -> ToolOutput:
        before_grid = inp.upstream_mask("before")
        before = before_grid.array.astype(bool)
        after = inp.upstream_mask("after").array.astype(bool)
        gain = clean_mask(after & ~before, params.min_region_px)
        loss = clean_mask(before & ~after, params.min_region_px)
        confs = [inp.upstream[k].confidence for k in ("before", "after")]
        known = [c for c in confs if c is not None]
        label = inp.upstream["before"].values.get("class", "target")
        n_before, n_after = int(before.sum()), int(after.sum())
        return ToolOutput(
            masks={
                "mask": before_grid.with_array(gain | loss),
                "gain": before_grid.with_array(gain),
                "loss": before_grid.with_array(loss),
                "before": before_grid.with_array(before),
                "after": before_grid.with_array(after),
            },
            values={
                "class": str(label),
                "before_pixels": n_before,
                "after_pixels": n_after,
                "gain_pixels": int(gain.sum()),
                "loss_pixels": int(loss.sum()),
                "net_change_pixels": n_after - n_before,
                "relative_change": (n_after - n_before) / n_before if n_before else None,
            },
            confidence=min(known) if known else None,
            confidence_basis="minimum confidence of the two class masks",
        )
