"""Shared synthetic BigEarthNet.txt fixtures (no real data, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("zstandard")
pytest.importorskip("tifffile")

from benvqa import testing as synth

SPLITS = {"train": 10, "validation": 4, "test": 3, "bench": 2}


@pytest.fixture(scope="session")
def ben_images() -> list[dict]:
    return synth.make_images(SPLITS, seed=0)


@pytest.fixture(scope="session")
def ben_parquet(tmp_path_factory: pytest.TempPathFactory, ben_images: list[dict]) -> Path:
    return synth.write_parquet(
        tmp_path_factory.mktemp("ann") / "BigEarthNet.txt.parquet", ben_images
    )


@pytest.fixture(scope="session")
def ben_tars(tmp_path_factory: pytest.TempPathFactory, ben_images: list[dict]) -> tuple[Path, Path]:
    return synth.write_tars(tmp_path_factory.mktemp("tars"), ben_images)
