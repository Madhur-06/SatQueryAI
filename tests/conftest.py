"""Shared fixtures: small synthetic GeoTIFFs with known ground truth.

All GeoTIFF scenes share one 128 x 128, 10 m grid in EPSG:32643 (see
``satquery.testing.synthetic``). They are written once per session; tests must treat
them as read-only and write anything new under ``tmp_path``.
"""

from pathlib import Path

import pytest

from satquery.testing.synthetic import (
    SceneGeometry,
    SyntheticRaster,
    write_optical,
    write_png,
    write_reprojected,
    write_sar,
    write_temporal_pair,
)


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
