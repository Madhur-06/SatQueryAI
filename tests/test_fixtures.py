"""Sanity checks on the synthetic fixtures: if these fail, every other test is suspect."""

import numpy as np
import rasterio
from rasterio.crs import CRS

from satquery.testing.synthetic import SyntheticRaster


def _read(path) -> tuple[np.ndarray, rasterio.profiles.Profile, dict, tuple]:
    with rasterio.open(path) as src:
        return src.read().astype(np.float64), src.profile, src.tags(), src.descriptions


def test_optical_4band_structure(optical_4band: SyntheticRaster) -> None:
    data, profile, tags, desc = _read(optical_4band.path)
    assert profile["count"] == 4
    assert profile["dtype"] == "uint16"
    assert CRS.from_user_input(profile["crs"]) == CRS.from_epsg(32643)
    assert profile["transform"].a == 10.0
    assert desc == ("blue", "green", "red", "nir")
    assert tags["ACQUISITION_DATE"] == "2024-01-15"
    # NDWI cleanly separates the water disk from land.
    green, nir = data[1], data[3]
    ndwi = (green - nir) / (green + nir)
    water = optical_4band.truth["water"]
    assert ndwi[water].min() > 0.3
    assert ndwi[~water].max() < -0.3


def test_sar_db_structure(sar_db: SyntheticRaster) -> None:
    data, profile, _, desc = _read(sar_db.path)
    assert profile["count"] == 2
    assert profile["dtype"] == "float32"
    assert desc == ("VV", "VH")
    vv = data[0]
    water = sar_db.truth["water"]
    # Values are in a plausible dB range and water is much darker (speckle overlaps a little).
    assert vv.min() > -45
    assert vv.max() < 10
    assert np.median(vv[~water]) - np.median(vv[water]) > 10


def test_sar_linear_is_positive_power(sar_linear: SyntheticRaster) -> None:
    data, *_ = _read(sar_linear.path)
    assert data.min() > 0
    assert data.max() < 10


def test_temporal_pair_builtup_appears(temporal_pair) -> None:
    t1, t2 = temporal_pair
    d1, p1, tags1, _ = _read(t1.path)
    d2, p2, tags2, _ = _read(t2.path)
    assert p1["count"] == p2["count"] == 6
    assert p1["transform"] == p2["transform"]
    assert tags1["ACQUISITION_DATE"] < tags2["ACQUISITION_DATE"]

    def ndbi(d: np.ndarray) -> np.ndarray:
        return (d[4] - d[3]) / (d[4] + d[3])

    gain = t2.truth["builtup_gain"]
    assert gain.sum() > 0
    assert ndbi(d1)[gain].max() < 0
    assert ndbi(d2)[gain].min() > 0
    # 25 % x 30 % of a 128 x 128 grid at 10 m.
    assert t2.truth_hectares("builtup_gain") == gain.sum() * 100 / 10_000


def test_failure_case_fixtures(optical_no_crs, optical_disjoint, sar_other_crs, png_rgb) -> None:
    with rasterio.open(optical_no_crs.path) as src:
        assert src.crs is None
    with rasterio.open(optical_disjoint.path) as src:
        assert src.bounds.left == 550_000
    with rasterio.open(sar_other_crs) as src:
        assert src.crs == CRS.from_epsg(32644)
    with rasterio.open(png_rgb.path) as src:
        assert src.driver == "PNG"
        assert src.count == 3
