from pathlib import Path

import numpy as np
import pytest
import rasterio

from satquery.io.validate import InputBundle, validate_inputs
from satquery.testing.synthetic import builtup_mask, write_array
from satquery.tools.registry import Registry, default_registry


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
