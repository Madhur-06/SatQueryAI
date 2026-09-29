"""Footprints, overlap and pixel areas."""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Geod
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

Bounds = tuple[float, float, float, float]  # left, bottom, right, top

_GEOD = Geod(ellps="WGS84")


def crs_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or CRS.from_user_input(a) == CRS.from_user_input(b)


def bounds_in(bounds: Bounds, src_crs: str, dst_crs: str) -> Bounds:
    if crs_equal(src_crs, dst_crs):
        return bounds
    return tuple(transform_bounds(src_crs, dst_crs, *bounds, densify_pts=21))  # type: ignore[return-value]


def bounds_iou(a: Bounds, b: Bounds) -> float:
    """Intersection-over-union of two axis-aligned boxes in the same CRS."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crs_linear_unit_m(crs: str) -> float | None:
    """Metres per CRS unit for projected CRSs, None for geographic CRSs."""
    c = CRS.from_user_input(crs)
    if c.is_geographic:
        return None
    axis = c.axis_info[0]
    return float(axis.unit_conversion_factor) if axis.unit_conversion_factor else 1.0


def pixel_area_m2(transform: Affine, crs: str | None, height: int) -> np.ndarray | None:
    """Area of one pixel in m², per row (shape ``(height, 1)`` for broadcasting).

    Projected CRS: constant ``|a*e - b*d|`` scaled to metres. (Projection area distortion,
    e.g. UTM scale factor, is < 0.1 % inside a zone and ignored.)
    Geographic CRS: geodesic area on the WGS84 ellipsoid, computed per row.
    No CRS: ``None`` (areas can only be reported in pixels).
    """
    if crs is None:
        return None
    unit = crs_linear_unit_m(crs)
    if unit is not None:
        area = abs(transform.a * transform.e - transform.b * transform.d) * unit**2
        return np.full((height, 1), area)
    areas = np.empty((height, 1))
    for row in range(height):
        corners = [transform @ (0, row), transform @ (1, row), transform @ (1, row + 1)]
        corners.append(transform @ (0, row + 1))
        lons, lats = zip(*corners, strict=True)
        area, _ = _GEOD.polygon_area_perimeter(lons, lats)
        areas[row, 0] = abs(area)
    return areas
