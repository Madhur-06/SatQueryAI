"""Input validation: turn 1-2 uploaded files into one well-defined ``InputBundle``.

Resolves the input configuration (SINGLE_OPT, SINGLE_SAR, PAIR_OPT_SAR, PAIR_TEMPORAL),
checks georeferencing and overlap for pairs, and resamples the second image onto
the reference grid. The reference is the first layer in canonical order: the optical
image for opt-SAR pairs and the earlier date for temporal pairs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rasterio.transform import Affine

from satquery.config import get_settings
from satquery.errors import ErrorCode, SatQueryError
from satquery.io.align import same_grid, warp_to_grid
from satquery.io.bands import BandSource, resolve_bands
from satquery.io.geometry import bounds_in, bounds_iou, crs_equal
from satquery.io.georaster import GeoRaster, Transform
from satquery.io.modality import detect_modality
from satquery.io.preprocess import sar as sar_pp
from satquery.io.raster import RasterMeta, inspect_raster, read_raster
from satquery.types import LAYER_KEYS, BandRole, InputConfig, Modality, TaskType

OPT_SAR_MAX_DATE_GAP_DAYS = 12
LARGE_RES_RATIO = 4.0


class InputOverrides(BaseModel):
    """Per-file user overrides. ``band_map`` uses 1-based band indices (GDAL convention)."""

    model_config = ConfigDict(extra="forbid")

    modality: Modality | None = None
    acquired: date | None = None
    band_map: dict[BandRole, int] | None = None


class RasterLayer(BaseModel):
    """One input image on the bundle's reference grid.

    ``data`` is float32 ``(bands, H, W)``: optical values as stored in the file; SAR is
    always converted to dB (``units == "db"``) whatever the input scale.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    meta: RasterMeta
    modality: Modality
    modality_source: Literal["detected", "override"]
    modality_confidence: float
    modality_reasons: list[str]
    band_map: dict[BandRole, int]
    band_sources: dict[BandRole, BandSource]
    acquired: date | None
    acquired_source: Literal["tag", "filename", "override"] | None
    units: Literal["as_stored", "db"]
    sar_input_scale: sar_pp.SarScale | None = None
    resampled: bool = False
    data: np.ndarray = Field(repr=False)
    valid: np.ndarray = Field(repr=False)
    transform: Transform
    crs: str | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[2]

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)

    def has(self, *roles: BandRole) -> bool:
        return all(r in self.band_map for r in roles)

    def band(self, role: BandRole) -> np.ndarray:
        if role not in self.band_map:
            raise SatQueryError(
                ErrorCode.BAND_ROLE_MISSING,
                f"{self.meta.filename}: no '{role}' band could be identified.",
                details={"file": self.meta.filename, "role": role, "known": list(self.band_map)},
                hint=f"Pass a band_map override, e.g. {{'{role}': <1-based band index>}}.",
            )
        return self.data[self.band_map[role]]

    def georaster(self, array: np.ndarray, nodata: float | None = None) -> GeoRaster:
        """Wrap a derived array (mask, index...) in this layer's grid."""
        return GeoRaster(array=array, transform=self.transform, crs=self.crs, nodata=nodata)

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "filename": self.meta.filename,
            "sha256": self.meta.sha256,
            "driver": self.meta.driver,
            "modality": self.modality,
            "modality_source": self.modality_source,
            "modality_confidence": round(self.modality_confidence, 3),
            "modality_reasons": self.modality_reasons,
            "bands": self.meta.count,
            "dtype": self.meta.dtype,
            "band_map": {str(k): v + 1 for k, v in self.band_map.items()},  # 1-based for humans
            "band_sources": {str(k): v for k, v in self.band_sources.items()},
            "units": self.units,
            "sar_input_scale": self.sar_input_scale,
            "crs": self.meta.crs,
            "resolution": list(self.meta.res),
            "native_shape": [self.meta.height, self.meta.width],
            "bounds": list(self.meta.bounds),
            "acquired": self.acquired.isoformat() if self.acquired else None,
            "acquired_source": self.acquired_source,
            "resampled": self.resampled,
            "valid_fraction": round(float(self.valid.mean()), 4),
        }


class AlignmentReport(BaseModel):
    crs_match: bool
    footprint_iou: float
    res_ratio: float
    """Coarser / finer pixel size of the two inputs (>= 1)."""
    resampled: bool
    method: str | None


class InputBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: InputConfig
    layers: dict[str, RasterLayer]
    alignment: AlignmentReport | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def reference(self) -> RasterLayer:
        return next(iter(self.layers.values()))

    def layer(self, key: str) -> RasterLayer:
        try:
            return self.layers[key]
        except KeyError:
            raise KeyError(f"bundle ({self.config}) has no layer '{key}'") from None

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "layers": [layer.summary() for layer in self.layers.values()],
            "alignment": self.alignment.model_dump() if self.alignment else None,
            "grid": {
                "crs": self.reference.crs,
                "transform": list(self.reference.transform),
                "shape": list(self.reference.shape),
            },
        }


# --------------------------------------------------------------------------- loading


def _load_layer(path: Path, override: InputOverrides, benchmark: bool) -> RasterLayer:
    meta = inspect_raster(path, benchmark=benchmark)
    data, valid = read_raster(meta)

    decision = detect_modality(meta, data, valid)
    if override.modality is not None:
        modality, source = override.modality, "override"
        confidence, reasons = 1.0, [f"set by user (detector said {decision.modality})"]
    else:
        modality, source = decision.modality, "detected"
        confidence, reasons = decision.confidence, decision.reasons
    if modality is Modality.UNKNOWN:
        raise SatQueryError(
            ErrorCode.MODALITY_UNKNOWN,
            f"{meta.filename}: could not tell whether this is optical or SAR imagery.",
            details={"file": meta.filename, "reasons": decision.reasons},
            hint="Set the modality explicitly (optical, multispectral or sar).",
        )

    bands = resolve_bands(meta, override.band_map)

    acquired, acquired_source = meta.acquired, meta.acquired_source
    if override.acquired is not None:
        acquired, acquired_source = override.acquired, "override"

    units: Literal["as_stored", "db"] = "as_stored"
    scale = None
    if modality is Modality.SAR:
        scale = sar_pp.detect_scale(data, valid, meta.dtype)
        if scale != "db":
            data = np.where(valid, sar_pp.to_db(sar_pp.to_linear_power(data, scale)), 0.0)
        units = "db"

    return RasterLayer(
        key="",
        meta=meta,
        modality=modality,
        modality_source=source,
        modality_confidence=confidence,
        modality_reasons=reasons,
        band_map=bands.band_map,
        band_sources=bands.sources,
        acquired=acquired,
        acquired_source=acquired_source,
        units=units,
        sar_input_scale=scale,
        data=data.astype(np.float32),
        valid=valid,
        transform=meta.transform,
        crs=meta.crs,
    )


# --------------------------------------------------------------------------- pairs


def _order_pair(
    a: RasterLayer, b: RasterLayer, warnings: list[str]
) -> tuple[InputConfig, RasterLayer, RasterLayer]:
    if a.modality.family != b.modality.family:
        opt, sar = (a, b) if a.modality.family == "optical" else (b, a)
        if opt.acquired and sar.acquired:
            gap = abs((opt.acquired - sar.acquired).days)
            if gap > OPT_SAR_MAX_DATE_GAP_DAYS:
                warnings.append(
                    f"Optical and SAR were acquired {gap} days apart; joint analysis assumes "
                    "the scene did not change in between."
                )
        return InputConfig.PAIR_OPT_SAR, opt, sar

    if a.acquired and b.acquired:
        if a.acquired == b.acquired:
            warnings.append("Both images have the same acquisition date.")
        first, second = (a, b) if a.acquired <= b.acquired else (b, a)
    else:
        first, second = a, b
        warnings.append(
            "Acquisition date missing for at least one image; using upload order "
            "(first upload = earlier date). Set dates explicitly to be sure."
        )
    if a.modality.family == "optical":
        common = set(first.band_map) & set(second.band_map)
        if not common and first.meta.count != second.meta.count:
            raise SatQueryError(
                ErrorCode.INCOMPATIBLE_PAIR,
                "The two dates have different band counts and no identifiable common bands.",
                details={"counts": [first.meta.count, second.meta.count]},
                hint="Provide band_map overrides for both images.",
            )
        if first.meta.count != second.meta.count:
            warnings.append(
                f"Band counts differ ({first.meta.count} vs {second.meta.count}); "
                f"only common bands {sorted(common)} are comparable."
            )
    return InputConfig.PAIR_TEMPORAL, first, second


def _align_pair(
    ref: RasterLayer,
    other: RasterLayer,
    *,
    allow_reproject: bool,
    min_overlap_iou: float,
    warnings: list[str],
) -> tuple[RasterLayer, AlignmentReport]:
    names = [ref.meta.filename, other.meta.filename]

    if ref.crs is None or other.crs is None:
        both_benchmark = ref.meta.is_benchmark_format and other.meta.is_benchmark_format
        if both_benchmark and ref.shape == other.shape:
            warnings.append("No georeferencing: images are assumed to be pixel-aligned.")
            report = AlignmentReport(
                crs_match=False, footprint_iou=1.0, res_ratio=1.0, resampled=False,
                method="pixel grid (no CRS)",
            )  # fmt: skip
            return other, report
        missing = [n for n, layer in zip(names, (ref, other), strict=True) if layer.crs is None]
        raise SatQueryError(
            ErrorCode.MISSING_CRS,
            "Image pairs must be georeferenced so they can be aligned.",
            details={"files_without_crs": missing},
            hint="Export the images as GeoTIFF with a CRS.",
        )

    crs_match = crs_equal(ref.crs, other.crs)
    if not crs_match and not allow_reproject:
        raise SatQueryError(
            ErrorCode.CRS_MISMATCH,
            f"The images use different coordinate systems ({ref.crs} vs {other.crs}).",
            details={"files": names, "crs": [ref.crs, other.crs]},
            hint="Reproject one image to the other's CRS, or allow automatic reprojection.",
        )

    other_bounds = bounds_in(other.meta.bounds, other.crs, ref.crs)
    iou = bounds_iou(ref.meta.bounds, other_bounds)
    other_res = (other_bounds[2] - other_bounds[0]) / other.meta.width  # in ref CRS units
    res_ratio = max(other_res / ref.meta.res[0], ref.meta.res[0] / other_res)
    details = {"files": names, "footprint_iou": round(iou, 4), "min_overlap_iou": min_overlap_iou}
    if iou == 0.0:
        raise SatQueryError(
            ErrorCode.NO_OVERLAP,
            "The two images do not overlap.",
            details=details,
            hint="Upload two images of the same area.",
        )
    if iou < min_overlap_iou:
        raise SatQueryError(
            ErrorCode.LOW_OVERLAP,
            f"The images overlap too little (IoU {iou:.2f} < {min_overlap_iou}).",
            details=details,
            hint="Crop both images to their common area.",
        )
    if res_ratio > LARGE_RES_RATIO:
        warnings.append(f"Pixel sizes differ by a factor of {res_ratio:.1f}.")

    if crs_match and same_grid(ref.affine, ref.shape, other.affine, other.shape):
        report = AlignmentReport(
            crs_match=True, footprint_iou=iou, res_ratio=res_ratio, resampled=False, method=None
        )
        return other, report

    # SAR is resampled in linear power, not dB, so averaging is physically meaningful.
    data = sar_pp.to_linear(other.data) if other.units == "db" else other.data
    warped, valid = warp_to_grid(
        data,
        other.valid,
        src_transform=other.affine,
        src_crs=other.crs,
        dst_transform=ref.affine,
        dst_crs=ref.crs,
        dst_shape=ref.shape,
    )
    if other.units == "db":
        warped = np.where(valid, sar_pp.to_db(warped), 0.0).astype(np.float32)
    action = "reprojected" if not crs_match else "resampled"
    warnings.append(f"{other.meta.filename} was {action} onto the grid of {ref.meta.filename}.")
    aligned = other.model_copy(
        update={
            "data": warped,
            "valid": valid,
            "transform": ref.transform,
            "crs": ref.crs,
            "resampled": True,
        }
    )
    report = AlignmentReport(
        crs_match=crs_match,
        footprint_iou=iou,
        res_ratio=res_ratio,
        resampled=True,
        method=f"bilinear{' (linear power)' if other.units == 'db' else ''}",
    )
    return aligned, report


# --------------------------------------------------------------------------- public API


def validate_inputs(
    paths: list[Path] | tuple[Path, ...],
    overrides: list[InputOverrides | None] | None = None,
    *,
    benchmark: bool = False,
    allow_reproject: bool = True,
    min_overlap_iou: float | None = None,
) -> InputBundle:
    """Validate 1-2 images and resolve them into an ``InputBundle``.

    Raises ``SatQueryError`` with a specific ``ErrorCode`` for every rejection.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise SatQueryError(ErrorCode.NO_INPUTS, "No input image was provided.")
    if len(paths) > 2:
        raise SatQueryError(
            ErrorCode.TOO_MANY_INPUTS,
            f"At most two images are supported, got {len(paths)}.",
            details={"count": len(paths)},
        )
    if overrides is not None and len(overrides) != len(paths):
        raise SatQueryError(
            ErrorCode.INVALID_OVERRIDE,
            "Provide one overrides entry per input image.",
            details={"images": len(paths), "overrides": len(overrides)},
        )
    min_iou = get_settings().min_overlap_iou if min_overlap_iou is None else min_overlap_iou
    per_file = overrides or [None] * len(paths)
    loaded = [
        _load_layer(path, o or InputOverrides(), benchmark)
        for path, o in zip(paths, per_file, strict=True)
    ]
    warnings: list[str] = []

    if len(loaded) == 1:
        layer = loaded[0]
        is_opt = layer.modality.family == "optical"
        config = InputConfig.SINGLE_OPT if is_opt else InputConfig.SINGLE_SAR
        if layer.crs is None:
            warnings.append(f"{layer.meta.filename} has no CRS: areas are reported in pixels only.")
        key = LAYER_KEYS[config][0]
        return InputBundle(
            config=config, layers={key: layer.model_copy(update={"key": key})}, warnings=warnings
        )

    config, first, second = _order_pair(loaded[0], loaded[1], warnings)
    second, report = _align_pair(
        first, second, allow_reproject=allow_reproject, min_overlap_iou=min_iou, warnings=warnings
    )
    k1, k2 = LAYER_KEYS[config]
    layers = {k1: first.model_copy(update={"key": k1}), k2: second.model_copy(update={"key": k2})}
    return InputBundle(config=config, layers=layers, alignment=report, warnings=warnings)


# Which input configurations each task can run on.
TASK_CONFIGS: dict[TaskType, frozenset[InputConfig]] = {
    TaskType.VQA: frozenset(InputConfig),
    TaskType.CAPTION: frozenset(InputConfig),
    TaskType.GROUNDING: frozenset(InputConfig),
    TaskType.CHANGE_VQA: frozenset({InputConfig.PAIR_TEMPORAL}),
    TaskType.CHANGE_DESC: frozenset({InputConfig.PAIR_TEMPORAL}),
    TaskType.FUSION_EXTRACT: frozenset({InputConfig.PAIR_OPT_SAR}),
}

_TASK_HINTS: dict[TaskType, str] = {
    TaskType.CHANGE_VQA: "Change questions need two images of the same area from different dates.",
    TaskType.CHANGE_DESC: "Change descriptions need two images of the same area from different "
    "dates.",
    TaskType.FUSION_EXTRACT: "Joint optical-SAR analysis needs one optical and one SAR image of "
    "the same area.",
}


def check_task_compatibility(task: TaskType, config: InputConfig) -> None:
    allowed = TASK_CONFIGS[task]
    if config in allowed:
        return
    raise SatQueryError(
        ErrorCode.TASK_INPUT_MISMATCH,
        _TASK_HINTS.get(task, f"Task '{task}' cannot run on input configuration '{config}'."),
        details={"task": task, "config": config, "supported_configs": sorted(allowed)},
        hint="Upload the matching images, or rephrase the question.",
    )
