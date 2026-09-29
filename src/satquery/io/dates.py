"""Acquisition-date extraction from raster tags and filenames."""

import re
from datetime import date, datetime

# Tag keys seen in GDAL/GeoTIFF exports (case-insensitive match).
DATE_TAG_KEYS = (
    "acquisition_date",
    "acquisitiondate",
    "date_acquired",
    "sensing_time",
    "sensing_date",
    "product_start_time",
    "start_time",
    "datetime",
    "date",
    "tifftag_datetime",
)

_ISO_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_:/.]?(0[1-9]|1[0-2])[-_:/.]?(0[1-9]|[12]\d|3[01])")


def _parse(text: str) -> date | None:
    for match in _ISO_RE.finditer(text):
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
    return None


def date_from_tags(tags: dict[str, str]) -> date | None:
    """First parseable date among well-known tag keys (in priority order)."""
    lowered = {k.lower(): v for k, v in tags.items()}
    for key in DATE_TAG_KEYS:
        if key in lowered and (found := _parse(lowered[key])):
            return found
    return None


def date_from_filename(name: str) -> date | None:
    """Date embedded in a filename, e.g. ``S2A_20240301_T43.tif`` or ``area_2024-03-01.tif``.

    Only 8-digit ``YYYYMMDD`` (optionally separated) is recognised, and it must not be
    part of a longer digit run, so tile numbers and hashes are not misread as dates.
    """
    stem = name.rsplit(".", 1)[0]
    for pattern in _FILENAME_DATE_PATTERNS:
        for match in pattern.finditer(stem):
            if found := _parse(match.group(0)):
                return found
    return None


_FILENAME_DATE_PATTERNS = (
    re.compile(r"(?<!\d)\d{8}(?!\d)"),  # 20240301 (also inside 20240301T052651)
    re.compile(r"(?<!\d)\d{4}[-_]\d{2}[-_]\d{2}(?!\d)"),  # 2024-03-01, 2024_03_01
)


def parse_iso_date(value: str) -> date:
    """Strict ISO date for API overrides (``YYYY-MM-DD``)."""
    return datetime.strptime(value, "%Y-%m-%d").date()
