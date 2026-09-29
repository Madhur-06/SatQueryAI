from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from satquery.io.geometry import bounds_iou, crs_equal, pixel_area_m2
from satquery.io.georaster import GeoRaster, to_transform
from satquery.io.raster import inspect_raster, read_raster
from satquery.io.render import blend, mask_rgba, quicklook, save_png
from satquery.io.tiling import iter_tiles, run_tiled, stitch
from satquery.io.vector import mask_to_geojson
from satquery.types import BandRole, Modality

UTM = to_transform(from_origin(500_000, 2_000_000, 10, 10))

# ------------------------------------------------------------------ geometry


def test_bounds_iou() -> None:
    assert bounds_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bounds_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert bounds_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_crs_equal() -> None:
    assert crs_equal("EPSG:4326", rasterio.crs.CRS.from_epsg(4326).to_wkt())
    assert not crs_equal("EPSG:32643", "EPSG:32644")
    assert crs_equal(None, None)
    assert not crs_equal("EPSG:4326", None)


def test_pixel_area_projected_and_geographic() -> None:
    projected = pixel_area_m2(Affine(*UTM), "EPSG:32643", 3)
    assert projected is not None
    assert projected.shape == (3, 1)
    assert projected[0, 0] == pytest.approx(100.0)

    deg = 1e-4
    at_equator = pixel_area_m2(from_origin(75.0, 0.0, deg, deg), "EPSG:4326", 1)
    at_60 = pixel_area_m2(from_origin(75.0, 60.0, deg, deg), "EPSG:4326", 1)
    assert at_equator is not None
    assert at_60 is not None
    assert at_equator[0, 0] == pytest.approx(111_320 * 110_574 * deg**2, rel=0.01)
    # At 60 N a degree of longitude is ~55.80 km, and a degree of latitude ~111.41 km
    # (longer than at the equator: ellipsoid flattening).
    assert at_60[0, 0] == pytest.approx(55_800 * 111_412 * deg**2, rel=0.005)
    assert pixel_area_m2(Affine(*UTM), None, 3) is None


# ------------------------------------------------------------------ tiling


@pytest.mark.parametrize(("shape", "size", "overlap"), [((100, 130), 32, 8), ((50, 50), 64, 8)])
def test_tile_cores_partition_the_image(shape, size, overlap) -> None:
    tiles = iter_tiles(shape, UTM, size=size, overlap=overlap)
    coverage = np.zeros(shape, dtype=int)
    for t in tiles:
        r, c = t.row_off + t.core_row0, t.col_off + t.core_col0
        coverage[r : t.row_off + t.core_row1, c : t.col_off + t.core_col1] += 1
        assert t.height <= size
        assert t.width <= size
        assert Affine(*t.transform) @ (0, 0) == Affine(*UTM) @ (t.col_off, t.row_off)
    assert (coverage == 1).all()


def test_run_tiled_reassembles_and_places_correctly() -> None:
    data = np.random.default_rng(0).random((2, 100, 130)).astype(np.float32)
    identity = run_tiled(lambda d, t: d, data, UTM, "EPSG:32643", size=32, overlap=8)
    np.testing.assert_array_equal(identity.array, data)
    assert identity.crs == "EPSG:32643"

    rows = run_tiled(
        lambda d, t: np.broadcast_to(np.arange(t.height)[:, None] + t.row_off, d.shape[-2:]),
        data,
        UTM,
        None,
        size=32,
        overlap=8,
    )
    np.testing.assert_array_equal(rows.array, np.broadcast_to(np.arange(100)[:, None], (100, 130)))


def test_tiling_errors() -> None:
    with pytest.raises(ValueError, match="overlap"):
        iter_tiles((10, 10), UTM, size=8, overlap=8)
    with pytest.raises(ValueError, match="no tiles"):
        stitch([], (10, 10), UTM, None)


# ------------------------------------------------------------------ georaster / vector / render


def test_georaster_geotiff_roundtrip(tmp_path: Path) -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    gr = GeoRaster(array=mask, transform=UTM, crs="EPSG:32643")
    path = gr.to_geotiff(tmp_path / "mask.tif")
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32643
        assert src.transform == Affine(*UTM)
        assert src.read(1).sum() == 16
    with pytest.raises(ValueError, match="does not match"):
        gr.with_array(np.zeros((4, 4)))


def test_mask_to_geojson_is_wgs84(optical_4band) -> None:
    water = optical_4band.truth["water"]
    fc = mask_to_geojson(
        GeoRaster(array=water, transform=UTM, crs="EPSG:32643"), properties={"class": "water"}
    )
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    feature = fc["features"][0]
    assert feature["properties"] == {"class": "water", "pixels": int(water.sum())}
    lon, lat = feature["geometry"]["coordinates"][0][0]
    # UTM 43N easting 500 km is the 75 E central meridian; northing 2000 km is ~18.1 N.
    assert lon == pytest.approx(75.0, abs=0.02)
    assert lat == pytest.approx(18.08, abs=0.02)


def test_mask_to_geojson_pixel_space_and_min_pixels() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:3, 1:3] = True  # 4 px
    mask[10:18, 10:18] = True  # 64 px
    fc = mask_to_geojson(GeoRaster(array=mask, transform=to_transform(Affine.identity()), crs=None))
    assert fc["satquery:coordinates"] == "pixel"
    assert [f["properties"]["pixels"] for f in fc["features"]] == [64, 4]
    small_dropped = mask_to_geojson(
        GeoRaster(array=mask, transform=UTM, crs="EPSG:32643"), min_pixels=10
    )
    assert len(small_dropped["features"]) == 1


def test_quicklooks_and_overlays(tmp_path: Path, optical_4band, sar_db) -> None:
    data, valid = read_raster(inspect_raster(optical_4band.path))
    bands = {BandRole.BLUE: 0, BandRole.GREEN: 1, BandRole.RED: 2, BandRole.NIR: 3}
    rgb = quicklook(data, valid, Modality.OPTICAL, bands)
    assert rgb.shape == (128, 128, 3)
    unknown = quicklook(data, valid, Modality.OPTICAL, {})
    assert unknown.shape == (128, 128, 3)

    sdata, svalid = read_raster(inspect_raster(sar_db.path))
    sar_rgb = quicklook(sdata, svalid, Modality.SAR, {BandRole.VV: 0, BandRole.VH: 1})
    water = sar_db.truth["water"]
    assert sar_rgb[~water].mean() > sar_rgb[water].mean()

    rgba = mask_rgba(water, (0, 0, 255), alpha=200)
    assert rgba.shape == (128, 128, 4)
    assert (rgba[water, 3] == 200).all()
    assert (rgba[~water, 3] == 0).all()
    blended = blend(rgb, water, (0, 0, 255), alpha=1.0)
    assert (blended[water] == (0, 0, 255)).all()

    path = save_png(rgba, tmp_path / "out" / "mask.png")
    with rasterio.open(path) as src:
        assert src.count == 4
