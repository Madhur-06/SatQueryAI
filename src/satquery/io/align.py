"""Warping a raster onto a reference grid."""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


def same_grid(
    a_transform: Affine,
    a_shape: tuple[int, int],
    b_transform: Affine,
    b_shape: tuple[int, int],
    tol: float = 1e-6,
) -> bool:
    """True if two grids (assumed to share a CRS) coincide pixel for pixel."""
    coeffs = zip(tuple(a_transform)[:6], tuple(b_transform)[:6], strict=True)
    return a_shape == b_shape and all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in coeffs)


def warp_to_grid(
    data: np.ndarray,
    valid: np.ndarray,
    *,
    src_transform: Affine,
    src_crs: str,
    dst_transform: Affine,
    dst_crs: str,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject ``(bands, H, W)`` data onto a target grid.

    Invalid source pixels are excluded (NaN during resampling). Returns the warped data
    and a validity mask; pixels outside the source footprint are invalid.
    """
    src = np.where(valid[np.newaxis], data, np.nan).astype(np.float32)
    out = np.full((data.shape[0], *dst_shape), np.nan, dtype=np.float32)
    for b in range(data.shape[0]):
        reproject(
            source=src[b],
            destination=out[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    out_valid = np.all(np.isfinite(out), axis=0)
    return np.where(out_valid[np.newaxis], out, 0.0).astype(np.float32), out_valid
