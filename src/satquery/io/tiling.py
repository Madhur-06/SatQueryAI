"""Tile large rasters with overlap and stitch per-tile outputs into a georeferenced mosaic.

Stitching uses "centre cropping": each output pixel comes from the tile in which it
lies furthest from a tile edge, which hides model edge artefacts without blending.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from pydantic import BaseModel
from rasterio.transform import Affine

from satquery.io.georaster import GeoRaster, Transform, to_transform


class Tile(BaseModel):
    row_off: int
    col_off: int
    height: int
    width: int
    transform: Transform
    """Geotransform of this tile (origin shifted to the tile's corner)."""
    # Core region (tile-local) that this tile contributes to the mosaic.
    core_row0: int
    core_row1: int
    core_col0: int
    core_col1: int

    def slice(self, array: np.ndarray) -> np.ndarray:
        """This tile's window of a ``(..., H, W)`` array."""
        return array[..., self.row_off : self.row_off + self.height,
                     self.col_off : self.col_off + self.width]  # fmt: skip


def _starts(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)  # last tile flush with the edge
    return sorted(set(starts))


def _core(starts: list[int], i: int, size: int, length: int) -> tuple[int, int]:
    """Tile-local [c0, c1) half-way between neighbouring tiles' overlaps."""
    start = starts[i]
    end = min(start + size, length)
    lo = 0 if i == 0 else (starts[i - 1] + size + start) // 2 - start
    hi = end - start if i == len(starts) - 1 else (end + starts[i + 1]) // 2 - start
    return lo, hi


def iter_tiles(
    shape: tuple[int, int], transform: Transform, size: int = 1024, overlap: int = 64
) -> list[Tile]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    height, width = shape
    affine = Affine(*transform)
    step = size - overlap
    rows, cols = _starts(height, size, step), _starts(width, size, step)
    tiles = []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            h, w = min(size, height - r), min(size, width - c)
            r0, r1 = _core(rows, ri, size, height)
            c0, c1 = _core(cols, ci, size, width)
            tiles.append(
                Tile(
                    row_off=r,
                    col_off=c,
                    height=h,
                    width=w,
                    transform=to_transform(affine @ Affine.translation(c, r)),
                    core_row0=r0,
                    core_row1=r1,
                    core_col0=c0,
                    core_col1=c1,
                )
            )
    return tiles


def stitch(
    pieces: Iterable[tuple[Tile, np.ndarray]],
    shape: tuple[int, int],
    transform: Transform,
    crs: str | None,
    nodata: float | None = None,
) -> GeoRaster:
    """Assemble ``(tile, output)`` pairs (outputs are ``(..., h, w)``) into one raster."""
    out: np.ndarray | None = None
    for tile, arr in pieces:
        if arr.shape[-2:] != (tile.height, tile.width):
            raise ValueError(f"tile output {arr.shape[-2:]} != tile {(tile.height, tile.width)}")
        if out is None:
            fill = 0 if nodata is None else nodata
            out = np.full((*arr.shape[:-2], *shape), fill, dtype=arr.dtype)
        core = arr[..., tile.core_row0 : tile.core_row1, tile.core_col0 : tile.core_col1]
        r, c = tile.row_off + tile.core_row0, tile.col_off + tile.core_col0
        out[..., r : r + core.shape[-2], c : c + core.shape[-1]] = core
    if out is None:
        raise ValueError("no tiles to stitch")
    return GeoRaster(array=out, transform=transform, crs=crs, nodata=nodata)


def run_tiled(
    fn: Callable[[np.ndarray, Tile], np.ndarray],
    data: np.ndarray,
    transform: Transform,
    crs: str | None,
    size: int = 1024,
    overlap: int = 64,
) -> GeoRaster:
    """Apply ``fn(tile_data, tile)`` over tiles of ``(..., H, W)`` data and stitch the results."""
    tiles = iter_tiles(data.shape[-2:], transform, size, overlap)
    return stitch(((t, fn(t.slice(data), t)) for t in tiles), data.shape[-2:], transform, crs)
