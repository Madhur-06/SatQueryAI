"""Shared fixtures: small synthetic GeoTIFFs with known ground truth.

All GeoTIFF scenes share one 128 x 128, 10 m grid in EPSG:32643 (see
``satquery.testing.synthetic``). They are written once per session; tests must treat
them as read-only and write anything new under ``tmp_path``.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from satquery.io.validate import InputBundle, validate_inputs
from satquery.testing.synthetic import (
    SceneGeometry,
    SyntheticRaster,
    builtup_mask,
    write_array,
    write_optical,
    write_png,
    write_reprojected,
    write_sar,
    write_temporal_pair,
)
from satquery.tools.registry import Registry, default_registry


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-ml", action="store_true", help="run tests that load model weights")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-ml"):
        return
    skip = pytest.mark.skip(reason="loads model weights; run with --run-ml (just test-ml)")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("synthetic")


@pytest.fixture(scope="session")
def optical_4band(synth_dir: Path) -> SyntheticRaster:
    """B, G, R, NIR uint16 reflectance with a low-NIR water disk."""
    return write_optical(synth_dir / "optical_4band.tif", n_bands=4)


@pytest.fixture(scope="session")
def optical_6band(synth_dir: Path) -> SyntheticRaster:
    """B, G, R, NIR, SWIR1, SWIR2 single date (water + vegetation)."""
    return write_optical(synth_dir / "optical_6band.tif", n_bands=6)


@pytest.fixture(scope="session")
def sar_db(synth_dir: Path) -> SyntheticRaster:
    """VV/VH float32 sigma0 in dB, speckled, low-backscatter water; co-registered with optical."""
    return write_sar(synth_dir / "sar_vv_vh_db.tif", scale="db")


@pytest.fixture(scope="session")
def sar_linear(synth_dir: Path) -> SyntheticRaster:
    """Same scene as ``sar_db`` but in linear power."""
    return write_sar(synth_dir / "sar_vv_vh_linear.tif", scale="linear")


@pytest.fixture(scope="session")
def temporal_pair(synth_dir: Path) -> tuple[SyntheticRaster, SyntheticRaster]:
    """6-band optical, 2020-03-01 and 2024-03-01; a built-up block appears in date 2."""
    return write_temporal_pair(synth_dir / "temporal")


@pytest.fixture(scope="session")
def sar_other_crs(synth_dir: Path, sar_db: SyntheticRaster) -> Path:
    """``sar_db`` warped to UTM 44N: overlaps the optical scene but needs reprojection."""
    return write_reprojected(sar_db.path, synth_dir / "sar_utm44.tif", "EPSG:32644")


@pytest.fixture(scope="session")
def optical_no_crs(synth_dir: Path) -> SyntheticRaster:
    return write_optical(
        synth_dir / "optical_no_crs.tif", geometry=SceneGeometry(crs=None), acquired=None
    )


@pytest.fixture(scope="session")
def optical_disjoint(synth_dir: Path) -> SyntheticRaster:
    """Same CRS as the other scenes but 50 km away: zero footprint overlap."""
    geometry = SceneGeometry().shifted(dx=50_000, dy=0)
    return write_optical(synth_dir / "optical_disjoint.tif", geometry=geometry)


@pytest.fixture(scope="session")
def png_rgb(synth_dir: Path) -> SyntheticRaster:
    """Ungeoreferenced 8-bit RGB PNG (benchmark-style input)."""
    return write_png(synth_dir / "benchmark_rgb.png")


# ---------------------------------------------------------------- validated bundles + registry


@pytest.fixture(scope="session")
def registry() -> Registry:
    return default_registry()


@pytest.fixture(scope="session")
def opt_bundle(optical_4band) -> InputBundle:
    return validate_inputs([optical_4band.path])


@pytest.fixture(scope="session")
def sar_bundle(sar_db) -> InputBundle:
    return validate_inputs([sar_db.path])


@pytest.fixture(scope="session")
def optsar_bundle(optical_4band, sar_db) -> InputBundle:
    return validate_inputs([optical_4band.path, sar_db.path])


@pytest.fixture(scope="session")
def temporal_bundle(temporal_pair) -> InputBundle:
    return validate_inputs([p.path for p in temporal_pair])


@pytest.fixture(scope="session")
def sar_temporal_bundle(synth_dir: Path, sar_db) -> tuple[InputBundle, np.ndarray]:
    """SAR pair where new buildings (+8 dB VV/VH) appear in the built-up block at date 2."""
    with rasterio.open(sar_db.path) as src:
        data = src.read()
    block = builtup_mask(data.shape[1])
    data[:, block] += 8.0
    t2 = write_array(
        synth_dir / "sar_20240301.tif", data, band_names=("VV", "VH"), tags={"DATE": "2024-03-01"}
    )
    return validate_inputs([sar_db.path, t2]), block
