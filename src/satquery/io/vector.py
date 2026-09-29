"""Mask vectorisation to GeoJSON.

RFC 7946 requires WGS84 longitude/latitude, so polygons are traced in the raster's
native CRS and then reprojected to EPSG:4326. Masks without a CRS are exported in
pixel coordinates (column, row) and flagged with ``"satquery:coordinates": "pixel"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom

from satquery.io.georaster import GeoRaster


def _ring_area(ring: list) -> float:
    xs, ys = np.asarray(ring, dtype=float).T
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def mask_to_geojson(
    mask: GeoRaster,
    *,
    properties: dict[str, Any] | None = None,
    min_pixels: int = 1,
    max_features: int = 5000,
) -> dict[str, Any]:
    """FeatureCollection of the ``True`` regions of a 2-D boolean mask.

    Regions smaller than ``min_pixels`` are dropped; at most ``max_features`` of the
    largest regions are kept so outputs stay small for the UI.
    """
    arr = np.asarray(mask.array).astype(bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2-D")
    georeferenced = mask.crs is not None
    transform = mask.affine if georeferenced else Affine.identity()
    polys: list[tuple[float, dict]] = []
    for geom, _ in shapes(arr.astype(np.uint8), mask=arr, connectivity=4, transform=transform):
        outer, *holes = geom["coordinates"]
        area = _ring_area(outer) - sum(_ring_area(h) for h in holes)
        pixel_area = area / abs(transform.a * transform.e)
        if pixel_area >= min_pixels:
            polys.append((pixel_area, geom))
    polys.sort(key=lambda p: -p[0])
    truncated = len(polys) > max_features
    features = []
    for i, (npx, geom) in enumerate(polys[:max_features]):
        if georeferenced:
            geom = transform_geom(mask.crs, "EPSG:4326", geom, precision=7)
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": geom,
                "properties": {**(properties or {}), "pixels": round(npx)},
            }
        )
    collection: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if not georeferenced:
        collection["satquery:coordinates"] = "pixel"
    if truncated:
        collection["satquery:truncated"] = True
    return collection
