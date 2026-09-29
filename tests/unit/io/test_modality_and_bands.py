from pathlib import Path

import numpy as np
import pytest

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.bands import resolve_bands
from satquery.io.modality import detect_modality
from satquery.io.raster import inspect_raster, read_raster
from satquery.testing.synthetic import write_array, write_optical, write_sar
from satquery.types import BandRole, Modality


def _detect(path: Path, benchmark: bool = False):
    meta = inspect_raster(path, benchmark=benchmark)
    data, valid = read_raster(meta)
    return detect_modality(meta, data, valid)


# ------------------------------------------------------------------ modality


def test_modality_from_descriptions(optical_4band, optical_6band, sar_db) -> None:
    assert _detect(optical_4band.path).modality is Modality.OPTICAL
    assert _detect(optical_6band.path).modality is Modality.MULTISPECTRAL
    decision = _detect(sar_db.path)
    assert decision.modality is Modality.SAR
    assert decision.confidence >= 0.9


def test_modality_from_statistics_without_descriptions(tmp_path: Path) -> None:
    opt = write_optical(tmp_path / "o.tif", band_descriptions=False)
    sar_db = write_sar(tmp_path / "s_db.tif", band_descriptions=False)
    sar_lin = write_sar(tmp_path / "s_lin.tif", scale="linear", band_descriptions=False)

    assert _detect(opt.path).modality is Modality.OPTICAL
    db = _detect(sar_db.path)
    assert db.modality is Modality.SAR
    assert "dB" in db.reasons[0]
    lin = _detect(sar_lin.path)
    assert lin.modality is Modality.SAR
    assert "speckle" in lin.reasons[0]


def test_modality_from_sensor_tag(tmp_path: Path) -> None:
    data = np.random.default_rng(0).uniform(0.1, 0.2, (2, 32, 32)).astype(np.float32)
    path = write_array(tmp_path / "s1.tif", data, tags={"MISSION": "SENTINEL-1A"})
    assert _detect(path).modality is Modality.SAR


def test_modality_unknown_when_no_evidence(tmp_path: Path) -> None:
    data = 0.5 + np.random.default_rng(0).normal(0, 0.001, (2, 32, 32)).astype(np.float32)
    path = write_array(tmp_path / "mystery.tif", data)
    assert _detect(path).modality is Modality.UNKNOWN


def test_png_is_optical(png_rgb) -> None:
    assert _detect(png_rgb.path, benchmark=True).modality is Modality.OPTICAL


# ------------------------------------------------------------------ bands


def _bands(path: Path, override=None, benchmark: bool = False):
    return resolve_bands(inspect_raster(path, benchmark=benchmark), override)


def test_bands_from_descriptions(optical_6band, sar_db) -> None:
    res = _bands(optical_6band.path)
    assert res.band_map == {
        BandRole.BLUE: 0,
        BandRole.GREEN: 1,
        BandRole.RED: 2,
        BandRole.NIR: 3,
        BandRole.SWIR1: 4,
        BandRole.SWIR2: 5,
    }
    assert set(res.sources.values()) == {"description"}
    assert _bands(sar_db.path).band_map == {BandRole.VV: 0, BandRole.VH: 1}


@pytest.mark.parametrize(
    ("desc", "role"),
    [
        ("Near Infrared", BandRole.NIR),
        ("B8 (NIR)", BandRole.NIR),
        ("nir08", BandRole.NIR),
        ("swir16", BandRole.SWIR1),
        ("SWIR 2", BandRole.SWIR2),
        ("Red Edge 1", BandRole.RED_EDGE),
        ("Sigma0_VV_db", BandRole.VV),
        ("Red", BandRole.RED),
    ],
)
def test_description_aliases(tmp_path: Path, desc: str, role: BandRole) -> None:
    path = write_array(tmp_path / "b.tif", np.ones((1, 8, 8), np.float32), band_names=(desc,))
    assert _bands(path).band_map == {role: 0}


def test_no_positional_default(tmp_path: Path) -> None:
    opt = write_optical(tmp_path / "o.tif", band_descriptions=False)
    assert _bands(opt.path).band_map == {}


def test_override_is_one_based_and_wins(tmp_path: Path) -> None:
    opt = write_optical(tmp_path / "o.tif", band_descriptions=False)
    res = _bands(opt.path, {BandRole.GREEN: 2, BandRole.NIR: 4})
    assert res.band_map == {BandRole.GREEN: 1, BandRole.NIR: 3}
    assert res.sources[BandRole.NIR] == "override"


@pytest.mark.parametrize("override", [{BandRole.NIR: 5}, {BandRole.NIR: 0}, {"nir": 1, "red": 1}])
def test_invalid_override(optical_4band, override) -> None:
    with pytest.raises(SatQueryError) as exc:
        _bands(optical_4band.path, override)
    assert exc.value.code is ErrorCode.INVALID_OVERRIDE


def test_sensor_band_codes_need_sensor_tag(tmp_path: Path) -> None:
    data = np.ones((4, 8, 8), np.uint16)
    codes = ("B02", "B03", "B04", "B08")
    anon = write_array(tmp_path / "anon.tif", data, band_names=codes)
    assert _bands(anon).band_map == {}  # "B8" alone is ambiguous

    s2 = write_array(
        tmp_path / "s2.tif", data, band_names=codes, tags={"SPACECRAFT_NAME": "Sentinel-2A"}
    )
    assert _bands(s2).band_map[BandRole.NIR] == 3

    l8 = write_array(
        tmp_path / "l8.tif",
        data,
        band_names=("SR_B2", "SR_B3", "SR_B4", "B8"),
        tags={"SPACECRAFT_ID": "LANDSAT_8"},
    )
    res = _bands(l8)
    assert res.band_map[BandRole.PAN] == 3
    assert BandRole.NIR not in res.band_map


def test_bands_from_wavelength_tags(tmp_path: Path) -> None:
    data = np.ones((2, 8, 8), np.uint16)
    path = write_array(
        tmp_path / "wl.tif", data, band_tags=[{"wavelength": "0.56"}, {"WAVELENGTH_NM": "842"}]
    )
    assert _bands(path).band_map == {BandRole.GREEN: 0, BandRole.NIR: 1}


def test_bands_from_png_colour_interp(png_rgb) -> None:
    res = _bands(png_rgb.path, benchmark=True)
    assert res.band_map == {BandRole.RED: 0, BandRole.GREEN: 1, BandRole.BLUE: 2}
