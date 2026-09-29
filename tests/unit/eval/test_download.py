"""Download helpers, exercised offline with file:// URLs."""

import hashlib
from pathlib import Path

import pytest

from satquery.eval.download import _find_pair_dirs, fetch


def _source(tmp_path: Path, data: bytes = b"satquery" * 1000) -> tuple[str, bytes]:
    src = tmp_path / "remote.bin"
    src.write_bytes(data)
    return src.resolve().as_uri(), data


def test_fetch_verifies_size_and_md5(tmp_path: Path) -> None:
    url, data = _source(tmp_path)
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    dest = fetch(url, tmp_path / "out" / "file.bin", size=len(data), md5=md5, log=lambda _: None)
    assert dest.read_bytes() == data
    assert not dest.with_name("file.bin.part").exists()


def test_fetch_skips_verified_existing_file(tmp_path: Path) -> None:
    url, data = _source(tmp_path)
    dest = tmp_path / "file.bin"
    dest.write_bytes(data)
    messages: list[str] = []
    fetch(url, dest, size=len(data), log=messages.append)
    assert "already present" in messages[0]


def test_fetch_rejects_corrupt_download(tmp_path: Path) -> None:
    url, _ = _source(tmp_path)
    with pytest.raises(OSError, match="verification failed"):
        fetch(url, tmp_path / "file.bin", md5="0" * 32, log=lambda _: None)
    assert not (tmp_path / "file.bin").exists()
    assert not (tmp_path / "file.bin.part").exists()


def test_find_pair_dirs(tmp_path: Path) -> None:
    assert _find_pair_dirs(tmp_path) is None
    (tmp_path / "raw" / "train" / "im1").mkdir(parents=True)
    (tmp_path / "raw" / "train" / "im2").mkdir()
    im1, im2 = _find_pair_dirs(tmp_path)
    assert im1.name == "im1"
    assert im2.parent == im1.parent
