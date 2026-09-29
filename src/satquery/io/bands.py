"""Band-role resolution: which band index is green, NIR, VV...

Precedence: explicit override > band descriptions > wavelength tags > colour
interpretation (RGB only) > sensor band-code tables (only when the sensor is named in
the tags). There is no positional default: an unnamed 4-band file could be RGBN or
BGRN, so unresolved roles stay unresolved and tools that need them fail with
``BAND_ROLE_MISSING``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.raster import RasterMeta
from satquery.types import BandRole

BandSource = Literal["override", "description", "wavelength", "color_interp", "sensor_table"]


class BandResolution(BaseModel):
    band_map: dict[BandRole, int]
    """Role -> 0-based band index."""
    sources: dict[BandRole, BandSource]


def _word(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z]){pattern}(?![a-z])", re.IGNORECASE)


# Order matters: more specific names first (red edge before red, swir1/2 before swir).
_DESCRIPTION_PATTERNS: list[tuple[BandRole, re.Pattern[str]]] = [
    (BandRole.RED_EDGE, _word(r"(red.?edge|rededge)")),
    (BandRole.SWIR1, _word(r"swir.?(1(?![0-9])|16)")),
    (BandRole.SWIR2, _word(r"swir.?(2(?![0-9])|22)")),
    (BandRole.SWIR1, _word(r"swir")),
    (BandRole.NIR, _word(r"(nir|near.?infra.?red)")),
    (BandRole.BLUE, _word(r"blue")),
    (BandRole.GREEN, _word(r"green")),
    (BandRole.RED, _word(r"red")),
    (BandRole.PAN, _word(r"(pan|panchromatic)")),
    (BandRole.VV, _word(r"vv")),
    (BandRole.VH, _word(r"vh")),
    (BandRole.HH, _word(r"hh")),
    (BandRole.HV, _word(r"hv")),
]

# Centre-wavelength ranges in nm (sensor-agnostic).
_WAVELENGTH_RANGES: list[tuple[BandRole, float, float]] = [
    (BandRole.BLUE, 430, 520),
    (BandRole.GREEN, 520, 600),
    (BandRole.RED, 620, 700),
    (BandRole.RED_EDGE, 700, 760),
    (BandRole.NIR, 760, 1000),
    (BandRole.SWIR1, 1500, 1800),
    (BandRole.SWIR2, 2000, 2400),
]

# Band codes only mean something once the sensor is known: "B8" is NIR on Sentinel-2
# but panchromatic on Landsat 8/9.
_SENSOR_TABLES: list[tuple[re.Pattern[str], dict[str, BandRole]]] = [
    (
        re.compile(r"sentinel.?2", re.IGNORECASE),
        {
            "B2": BandRole.BLUE,
            "B3": BandRole.GREEN,
            "B4": BandRole.RED,
            "B5": BandRole.RED_EDGE,
            "B8": BandRole.NIR,
            "B11": BandRole.SWIR1,
            "B12": BandRole.SWIR2,
        },
    ),
    (
        re.compile(r"landsat.?(8|9)", re.IGNORECASE),
        {
            "B2": BandRole.BLUE,
            "B3": BandRole.GREEN,
            "B4": BandRole.RED,
            "B5": BandRole.NIR,
            "B6": BandRole.SWIR1,
            "B7": BandRole.SWIR2,
            "B8": BandRole.PAN,
        },
    ),
]


def _band_code(desc: str) -> str | None:
    m = re.search(r"(?<![a-z0-9])(?:sr_)?b0*(\d{1,2}a?)(?![0-9])", desc, re.IGNORECASE)
    return f"B{m.group(1).upper()}" if m else None


def _wavelength_nm(tags: dict[str, str]) -> float | None:
    for key, value in tags.items():
        if "wavelength" not in key.lower():
            continue
        m = re.search(r"\d+(\.\d+)?", value)
        if m:
            v = float(m.group(0))
            return v * 1000 if v < 20 else v  # micrometres -> nm
    return None


def validate_override(band_map: dict[BandRole, int], count: int) -> dict[BandRole, int]:
    """Check a user band map (1-based, GDAL convention) and return it 0-based."""
    bad = {role: idx for role, idx in band_map.items() if not 1 <= idx <= count}
    if bad:
        raise SatQueryError(
            ErrorCode.INVALID_OVERRIDE,
            f"Band indices out of range 1..{count}: {bad}",
            details={"band_map": {str(k): v for k, v in band_map.items()}, "band_count": count},
            hint="Band indices are 1-based, as in GDAL/QGIS ('Band 1' is the first band).",
        )
    if len(set(band_map.values())) != len(band_map):
        raise SatQueryError(
            ErrorCode.INVALID_OVERRIDE,
            "The same band index is assigned to more than one role.",
            details={"band_map": {str(k): v for k, v in band_map.items()}},
        )
    return {role: idx - 1 for role, idx in band_map.items()}


def resolve_bands(meta: RasterMeta, override: dict[BandRole, int] | None = None) -> BandResolution:
    band_map: dict[BandRole, int] = {}
    sources: dict[BandRole, BandSource] = {}

    def assign(role: BandRole, idx: int, source: BandSource) -> None:
        if role not in band_map and idx not in band_map.values():
            band_map[role] = idx
            sources[role] = source

    if override:
        for role, idx in validate_override(override, meta.count).items():
            assign(role, idx, "override")

    for idx, desc in enumerate(meta.band_descriptions):
        if not desc:
            continue
        for role, pattern in _DESCRIPTION_PATTERNS:
            if pattern.search(desc):
                assign(role, idx, "description")
                break

    for idx, tags in enumerate(meta.band_tags):
        if (nm := _wavelength_nm(tags)) is not None:
            for role, lo, hi in _WAVELENGTH_RANGES:
                if lo <= nm < hi:
                    assign(role, idx, "wavelength")
                    break

    colour_roles = {"red": BandRole.RED, "green": BandRole.GREEN, "blue": BandRole.BLUE}
    for idx, ci in enumerate(meta.color_interp):
        if ci in colour_roles:
            assign(colour_roles[ci], idx, "color_interp")

    sensor_text = " ".join(meta.tags.values())
    for sensor_re, table in _SENSOR_TABLES:
        if sensor_re.search(sensor_text):
            for idx, desc in enumerate(meta.band_descriptions):
                code = _band_code(desc or "")
                if code in table:
                    assign(table[code], idx, "sensor_table")
            break

    return BandResolution(band_map=band_map, sources=sources)
