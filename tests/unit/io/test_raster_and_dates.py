from datetime import date
from pathlib import Path

import pytest

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.dates import date_from_filename, date_from_tags
from satquery.io.raster import inspect_raster, read_raster


def test_inspect_optical(optical_4band) -> None:
    meta = inspect_raster(optical_4band.path)
    assert meta.count == 4
    assert meta.dtype == "uint16"
    assert meta.crs == "EPSG:32643"
    assert not meta.is_geographic
    assert meta.res == (10.0, 10.0)
    assert meta.bounds == (500_000.0, 2_000_000.0 - 1280, 500_000.0 + 1280, 2_000_000.0)
    assert meta.band_descriptions == ["blue", "green", "red", "nir"]
    assert meta.acquired == date(2024, 1, 15)
    assert meta.acquired_source == "tag"
    assert len(meta.sha256) == 64
    assert meta.nodata is None


def test_read_raster_shapes(sar_db) -> None:
    data, valid = read_raster(inspect_raster(sar_db.path))
    assert data.shape == (2, 128, 128)
    assert data.dtype.name == "float32"
    assert valid.all()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("s2_20200301.tif", date(2020, 3, 1)),
        ("S2A_MSIL2A_20240301T052651_N0510_R105_T43QCA.tif", date(2024, 3, 1)),
        ("area_2024-03-01.tif", date(2024, 3, 1)),
        ("area_2024_03_01_v2.tif", date(2024, 3, 1)),
        ("tile_123456789.tif", None),  # 9-digit run is not a date
        ("x_20241399.tif", None),  # invalid month/day
        ("scene.tif", None),
    ],
)
def test_date_from_filename(name: str, expected: date | None) -> None:
    assert date_from_filename(name) == expected


def test_date_from_tags_priority_and_formats() -> None:
    assert date_from_tags({"TIFFTAG_DATETIME": "2023:05:17 10:11:12"}) == date(2023, 5, 17)
    tags = {"DATETIME": "2020-01-01", "ACQUISITION_DATE": "2021-02-02"}
    assert date_from_tags(tags) == date(2021, 2, 2)
    assert date_from_tags({"foo": "2020-01-01"}) is None


def test_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "scene.nc"
    path.write_bytes(b"not a raster")
    with pytest.raises(SatQueryError) as exc:
        inspect_raster(path)
    assert exc.value.code is ErrorCode.UNSUPPORTED_FORMAT


def test_png_requires_benchmark_flag(png_rgb) -> None:
    with pytest.raises(SatQueryError) as exc:
        inspect_raster(png_rgb.path)
    assert exc.value.code is ErrorCode.UNSUPPORTED_FORMAT
    assert "benchmark" in (exc.value.info.hint or "")
    meta = inspect_raster(png_rgb.path, benchmark=True)
    assert meta.driver == "PNG"
    assert meta.crs is None
    assert meta.color_interp == ["red", "green", "blue"]


def test_corrupt_tif_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"II*\x00garbage")
    with pytest.raises(SatQueryError) as exc:
        inspect_raster(path)
    assert exc.value.code is ErrorCode.UNREADABLE_FILE
