"""Raster inspection and reading (rasterio)."""

from __future__ import annotations

import hashlib
import warnings
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from pydantic import BaseModel
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.dates import date_from_filename, date_from_tags
from satquery.io.georaster import Transform, to_transform

GEOTIFF_DRIVERS = frozenset({"GTiff", "COG"})
BENCHMARK_DRIVERS = frozenset({"PNG", "JPEG"})
GEOTIFF_SUFFIXES = frozenset({".tif", ".tiff"})
BENCHMARK_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


class RasterMeta(BaseModel):
    """Everything we know about a file before looking at its pixels."""

    path: Path
    filename: str
    driver: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None
    """``EPSG:xxxx`` when the CRS has an authority code, otherwise WKT."""
    is_geographic: bool
    transform: Transform
    res: tuple[float, float]
    bounds: tuple[float, float, float, float]
    """(left, bottom, right, top) in ``crs`` units (pixel units when crs is None)."""
    nodata: float | None
    band_descriptions: list[str | None]
    color_interp: list[str]
    tags: dict[str, str]
    band_tags: list[dict[str, str]]
    acquired: date | None
    acquired_source: Literal["tag", "filename", "override"] | None
    sha256: str
    size_bytes: int

    @property
    def is_benchmark_format(self) -> bool:
        return self.driver in BENCHMARK_DRIVERS


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _check_suffix(path: Path, benchmark: bool) -> None:
    suffix = path.suffix.lower()
    if suffix in GEOTIFF_SUFFIXES:
        return
    if suffix in BENCHMARK_SUFFIXES:
        if benchmark:
            return
        raise SatQueryError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"{path.name}: PNG/JPEG inputs are only accepted for public benchmark datasets.",
            details={"file": path.name, "suffix": suffix},
            hint="Upload a GeoTIFF, or set benchmark=true for benchmark images.",
        )
    raise SatQueryError(
        ErrorCode.UNSUPPORTED_FORMAT,
        f"{path.name}: unsupported file type '{suffix}'.",
        details={"file": path.name, "suffix": suffix, "supported": [".tif", ".tiff"]},
        hint="Upload a GeoTIFF (.tif/.tiff).",
    )


def inspect_raster(path: Path, *, benchmark: bool = False) -> RasterMeta:
    """Read metadata only. Raises ``SatQueryError`` for unsupported or unreadable files."""
    path = Path(path)
    _check_suffix(path, benchmark)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            src = rasterio.open(path)
    except RasterioIOError as exc:
        raise SatQueryError(
            ErrorCode.UNREADABLE_FILE,
            f"{path.name}: could not be read as a raster.",
            details={"file": path.name, "reason": str(exc)},
        ) from exc
    with src:
        allowed = GEOTIFF_DRIVERS | (BENCHMARK_DRIVERS if benchmark else frozenset())
        if src.driver not in allowed:
            raise SatQueryError(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"{path.name}: format '{src.driver}' does not match an accepted type.",
                details={"file": path.name, "driver": src.driver, "allowed": sorted(allowed)},
            )
        crs = src.crs
        crs_str = None
        if crs is not None:
            epsg = crs.to_epsg()
            crs_str = f"EPSG:{epsg}" if epsg else crs.to_wkt()
        tags = {str(k): str(v) for k, v in src.tags().items()}
        acquired = date_from_tags(tags)
        source: Literal["tag", "filename"] | None = "tag" if acquired else None
        if acquired is None and (acquired := date_from_filename(path.name)):
            source = "filename"
        meta = RasterMeta(
            path=path,
            filename=path.name,
            driver=src.driver,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=crs_str,
            is_geographic=bool(crs is not None and crs.is_geographic),
            transform=to_transform(src.transform),
            res=(float(src.res[0]), float(src.res[1])),
            bounds=tuple(float(v) for v in src.bounds),  # type: ignore[arg-type]
            nodata=None if src.nodata is None else float(src.nodata),
            band_descriptions=list(src.descriptions),
            color_interp=[ci.name for ci in src.colorinterp],
            tags=tags,
            band_tags=[
                {str(k): str(v) for k, v in src.tags(b).items()} for b in range(1, src.count + 1)
            ],
            acquired=acquired,
            acquired_source=source,
            sha256="",
            size_bytes=path.stat().st_size,
        )
    return meta.model_copy(update={"sha256": sha256_file(path)})


def read_raster(meta: RasterMeta) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, valid)``: float32 ``(bands, H, W)`` and a bool ``(H, W)`` mask.

    A pixel is valid when every band is finite and differs from the nodata value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(meta.path) as src:
            data = src.read().astype(np.float32)
    valid = np.all(np.isfinite(data), axis=0)
    if meta.nodata is not None and not np.isnan(meta.nodata):
        valid &= np.all(data != meta.nodata, axis=0)
    return data, valid
