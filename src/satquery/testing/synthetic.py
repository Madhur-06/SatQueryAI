"""Small synthetic georeferenced scenes with known ground truth.

Used by the test fixtures and for demo data. Every scene shares one layout on a
128 x 128, 10 m grid in UTM 43N (EPSG:32643), so optical, SAR and temporal images
are co-registered by construction:

* a disk of open water (low NIR / low SWIR in optical, low backscatter in SAR);
* a rectangular block that becomes built-up in the second date of the temporal pair;
* vegetated land everywhere else.

Optical values are surface reflectance x 10000 stored as uint16 (the common L2A
convention); SAR values are float32 sigma0, in dB or linear power.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

DEFAULT_CRS = "EPSG:32643"
DEFAULT_ORIGIN = (500_000.0, 2_000_000.0)  # upper-left corner, UTM 43N (~18 N, 75 E)

OPTICAL_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
SAR_BANDS = ("VV", "VH")

# Reflectance x 10000 per class, in OPTICAL_BANDS order.
SPECTRA: dict[str, tuple[int, ...]] = {
    "vegetation": (400, 700, 500, 3000, 1800, 900),
    "water": (600, 500, 300, 150, 60, 40),
    "builtup": (1200, 1300, 1400, 2000, 2600, 2200),
}
# Mean sigma0 in dB per class, in SAR_BANDS order.
BACKSCATTER_DB: dict[str, tuple[float, float]] = {
    "vegetation": (-8.0, -15.0),
    "water": (-22.0, -28.0),
}


@dataclass(frozen=True)
class SceneGeometry:
    size: int = 128
    pixel: float = 10.0
    origin: tuple[float, float] = DEFAULT_ORIGIN
    crs: str | None = DEFAULT_CRS

    @property
    def transform(self) -> Affine:
        return from_origin(self.origin[0], self.origin[1], self.pixel, self.pixel)

    def shifted(self, dx: float, dy: float) -> SceneGeometry:
        return replace(self, origin=(self.origin[0] + dx, self.origin[1] + dy))


@dataclass
class SyntheticRaster:
    path: Path
    geometry: SceneGeometry
    acquired: date | None
    truth: dict[str, np.ndarray] = field(default_factory=dict)
    """Boolean ground-truth masks on the image grid, keyed by class name."""

    def truth_pixels(self, key: str) -> int:
        return int(self.truth[key].sum())

    def truth_hectares(self, key: str) -> float:
        return self.truth_pixels(key) * self.geometry.pixel**2 / 10_000


def water_mask(size: int) -> np.ndarray:
    rows, cols = np.mgrid[0:size, 0:size]
    centre_r, centre_c, radius = 0.65 * size, 0.30 * size, 0.18 * size
    return (rows - centre_r) ** 2 + (cols - centre_c) ** 2 <= radius**2


def builtup_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[int(0.15 * size) : int(0.40 * size), int(0.55 * size) : int(0.85 * size)] = True
    return mask


def _optical_array(
    size: int, n_bands: int, classes: dict[str, np.ndarray], gain: float, rng: np.random.Generator
) -> np.ndarray:
    data = np.empty((n_bands, size, size), dtype=np.float64)
    for b in range(n_bands):
        data[b] = SPECTRA["vegetation"][b]
        for name, mask in classes.items():
            data[b][mask] = SPECTRA[name][b]
    data *= gain
    data *= 1.0 + rng.normal(0.0, 0.03, data.shape)
    return np.clip(np.round(data), 1, 10_000).astype(np.uint16)


def write_array(
    path: Path,
    data: np.ndarray,
    geometry: SceneGeometry | None = None,
    *,
    band_names: tuple[str | None, ...] | None = None,
    tags: dict[str, str] | None = None,
    band_tags: list[dict[str, str]] | None = None,
    acquired: date | None = None,
) -> Path:
    """Write an arbitrary ``(bands, H, W)`` array as a GeoTIFF on ``geometry``'s grid."""
    geometry = geometry or SceneGeometry(size=data.shape[1])
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "dtype": data.dtype.name,
        "transform": geometry.transform,
    }
    if geometry.crs is not None:
        profile["crs"] = geometry.crs
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if band_names:
            for i, name in enumerate(band_names, start=1):
                if name:
                    dst.set_band_description(i, name)
        all_tags = dict(tags or {})
        if acquired is not None:
            all_tags["ACQUISITION_DATE"] = acquired.isoformat()
        if all_tags:
            dst.update_tags(**all_tags)
        for i, bt in enumerate(band_tags or [], start=1):
            dst.update_tags(i, **bt)
    return path


def write_optical(
    path: Path,
    *,
    n_bands: Literal[4, 6] = 4,
    geometry: SceneGeometry | None = None,
    builtup: bool = False,
    gain: float = 1.0,
    acquired: date | None = date(2024, 1, 15),
    band_descriptions: bool = True,
    seed: int = 0,
) -> SyntheticRaster:
    """Optical scene: 4 bands (B, G, R, NIR) or 6 bands (+ SWIR1, SWIR2).

    ``builtup=True`` paints the built-up block; ``gain`` simulates a radiometric
    difference between dates (what histogram matching should remove).
    """
    geometry = geometry or SceneGeometry()
    size = geometry.size
    classes = {"water": water_mask(size)}
    if builtup:
        classes["builtup"] = builtup_mask(size)
    data = _optical_array(size, n_bands, classes, gain, np.random.default_rng(seed))
    names = OPTICAL_BANDS[:n_bands] if band_descriptions else None
    write_array(path, data, geometry, band_names=names, acquired=acquired)
    truth = dict(classes)
    truth.setdefault("builtup", np.zeros((size, size), dtype=bool))
    return SyntheticRaster(path=path, geometry=geometry, acquired=acquired, truth=truth)


def write_sar(
    path: Path,
    *,
    geometry: SceneGeometry | None = None,
    scale: Literal["db", "linear"] = "db",
    looks: int = 4,
    acquired: date | None = date(2024, 1, 16),
    band_descriptions: bool = True,
    seed: int = 1,
) -> SyntheticRaster:
    """Dual-pol (VV, VH) sigma0 with multiplicative gamma speckle (``looks``-look intensity)."""
    geometry = geometry or SceneGeometry()
    size = geometry.size
    water = water_mask(size)
    rng = np.random.default_rng(seed)
    data = np.empty((2, size, size), dtype=np.float64)
    for b in range(2):
        mean_db = np.full((size, size), BACKSCATTER_DB["vegetation"][b])
        mean_db[water] = BACKSCATTER_DB["water"][b]
        speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=(size, size))
        data[b] = 10 ** (mean_db / 10) * speckle
    if scale == "db":
        data = 10 * np.log10(data)
    names = SAR_BANDS if band_descriptions else None
    write_array(path, data.astype(np.float32), geometry, band_names=names, acquired=acquired)
    return SyntheticRaster(path=path, geometry=geometry, acquired=acquired, truth={"water": water})


def write_temporal_pair(
    directory: Path, *, n_bands: Literal[4, 6] = 6, geometry: SceneGeometry | None = None
) -> tuple[SyntheticRaster, SyntheticRaster]:
    """Two dates of the same area; the built-up block appears in date 2.

    Dates live both in the GeoTIFF tags and in the filenames. Date 2 is 8 % brighter
    overall to exercise histogram matching.
    """
    t1 = write_optical(
        directory / "s2_20200301.tif",
        n_bands=n_bands,
        geometry=geometry,
        acquired=date(2020, 3, 1),
        seed=10,
    )
    t2 = write_optical(
        directory / "s2_20240301.tif",
        n_bands=n_bands,
        geometry=geometry,
        builtup=True,
        gain=1.08,
        acquired=date(2024, 3, 1),
        seed=11,
    )
    t2.truth["builtup_gain"] = t2.truth["builtup"] & ~t1.truth["builtup"]
    return t1, t2


def write_reprojected(src: Path, dst: Path, dst_crs: str) -> Path:
    """Warp ``src`` into ``dst_crs`` (keeps tags and band descriptions)."""
    with rasterio.open(src) as s:
        transform, width, height = calculate_default_transform(
            s.crs, dst_crs, s.width, s.height, *s.bounds
        )
        profile = s.profile | {
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        }
        with rasterio.open(dst, "w", **profile) as d:
            for b in range(1, s.count + 1):
                reproject(
                    source=rasterio.band(s, b),
                    destination=rasterio.band(d, b),
                    src_transform=s.transform,
                    src_crs=s.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                d.set_band_description(b, s.descriptions[b - 1])
            d.update_tags(**s.tags())
    return dst


def write_png(path: Path, *, size: int = 128, seed: int = 2) -> SyntheticRaster:
    """8-bit RGB PNG (no georeferencing), as found in public benchmark datasets."""
    water = water_mask(size)
    data = _optical_array(size, 3, {"water": water}, 1.0, np.random.default_rng(seed))
    rgb = np.clip(data[::-1].transpose(1, 2, 0) / 2000 * 255, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    geometry = SceneGeometry(size=size, pixel=1.0, origin=(0.0, 0.0), crs=None)
    return SyntheticRaster(path=path, geometry=geometry, acquired=None, truth={"water": water})
