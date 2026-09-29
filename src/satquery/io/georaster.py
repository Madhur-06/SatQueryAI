"""A georeferenced array: the unit every spatial output travels in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict, field_validator
from rasterio.transform import Affine

Transform = tuple[float, float, float, float, float, float]
"""Affine coefficients (a, b, c, d, e, f) as in ``rasterio.Affine``."""


def to_transform(affine: Affine) -> Transform:
    return (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f)


class GeoRaster(BaseModel):
    """2-D (H, W) or 3-D (bands, H, W) array plus its grid. ``crs`` is None for pixel-space data."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    array: np.ndarray
    transform: Transform
    crs: str | None
    nodata: float | None = None

    @field_validator("array")
    @classmethod
    def _check_ndim(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim not in (2, 3):
            raise ValueError(f"GeoRaster array must be 2-D or 3-D, got {v.ndim}-D")
        return v

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape[-2], self.array.shape[-1]

    def with_array(self, array: np.ndarray, nodata: float | None = None) -> GeoRaster:
        """Same grid, new data (e.g. a mask derived from this raster)."""
        if array.shape[-2:] != self.shape:
            raise ValueError(f"shape {array.shape[-2:]} does not match grid {self.shape}")
        return GeoRaster(array=array, transform=self.transform, crs=self.crs, nodata=nodata)

    def summary(self) -> dict[str, Any]:
        """JSON-safe description (no pixel data) for traces."""
        return {
            "shape": list(self.array.shape),
            "dtype": self.array.dtype.name,
            "crs": self.crs,
            "transform": list(self.transform),
        }

    def to_geotiff(self, path: Path) -> Path:
        data = self.array[np.newaxis] if self.array.ndim == 2 else self.array
        dtype = "uint8" if data.dtype == bool else data.dtype.name
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "width": data.shape[2],
            "height": data.shape[1],
            "count": data.shape[0],
            "dtype": dtype,
            "transform": self.affine,
            "compress": "deflate",
        }
        if self.crs is not None:
            profile["crs"] = self.crs
        if self.nodata is not None:
            profile["nodata"] = self.nodata
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype(dtype))
        return path
