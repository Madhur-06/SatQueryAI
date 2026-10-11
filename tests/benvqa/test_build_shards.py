"""Stage 0: quantization, selection, streaming tar -> shards, resume and HTTP drops."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from benvqa import testing as synth
from benvqa.data import build_shards as bs
from benvqa.data.httpstream import ParallelRangeReader, ResumableHTTPStream
from benvqa.data.quantize import (
    CHANNELS,
    S2_BANDS,
    QuantSpec,
    cube_from_bytes,
    dequantize_s1,
    dequantize_s2,
    encode_cube,
    quantize_s1,
    quantize_s2,
    upsample,
)
from benvqa.hub_sync import LocalMirror

SPEC = QuantSpec()


def test_s2_quantization_monotonic_and_sqrt_precise() -> None:
    dn = np.arange(0, 12000, dtype=np.uint16)
    u8 = quantize_s2(dn, "B02", SPEC)
    assert np.all(np.diff(u8.astype(int)) >= 0)
    assert u8[0] == 0
    assert u8[-1] == 255
    refl = dn / 10000
    err = np.abs(dequantize_s2(u8, "B02", SPEC) - refl)[dn < 3000]
    dark = err[:500]  # water-like reflectance < 0.05: sqrt encoding keeps it fine
    assert dark.max() < 0.0012
    assert err.max() < 0.0025


def test_s1_quantization_roundtrip_and_nan() -> None:
    db = np.array([-50, -30, -12.5, 0, 5, 20, np.nan], dtype=np.float32)
    u8 = quantize_s1(db, "VV", SPEC)
    assert list(u8[[0, 1, 4, 5, 6]]) == [0, 0, 255, 255, 0]
    back = dequantize_s1(u8, "VV", SPEC)
    assert abs(back[2] + 12.5) < 35 / 255


def test_upsample_preserves_constant_and_ramp() -> None:
    assert np.allclose(upsample(np.full((60, 60), 7.0)), 7.0)
    ramp = np.tile(np.arange(60, dtype=np.float32), (60, 1))
    up = upsample(ramp)
    assert up.shape == (120, 120)
    # pixel-centre aligned 2x: output column j samples input position (j + 0.5) / 2 - 0.5
    assert np.allclose(up[0, 2:-2], (np.arange(120)[2:-2] + 0.5) / 2 - 0.5, atol=1e-5)


def test_quant_spec_json_roundtrip() -> None:
    assert QuantSpec.from_json(SPEC.to_json()) == SPEC


def test_selection_never_includes_test(ben_parquet: Path) -> None:
    sel = bs.select_images(ben_parquet, val_small=2, seed=0)
    splits = sel.by_split()
    assert splits == {"train": 10, "val_small": 2, "bench": 2}
    rec = next(r for r in sel.images.values() if r.split == "train")
    qa = json.loads(rec.qa)
    assert {q["type"] for q in qa} == {"binary", "mcq", "captioning"}
    assert all(q["type"] == "bounding box" for q in json.loads(rec.bbox))
    assert all(q["category"] != "None" for q in qa)
    full = bs.select_images(ben_parquet, val_small=1, include_full_val=True)
    assert full.by_split() == {"train": 10, "val_small": 1, "val": 3, "bench": 2}


def _args(parquet: Path, s1: str, s2: str, mirror: Path, work: Path, **kw) -> Namespace:
    base = dict(
        parquet=parquet,
        mirror=f"local:{mirror}",
        private=False,
        work_dir=work,
        s1_url=s1,
        s2_url=s2,
        val_small=2,
        include_full_val=False,
        seed=0,
        shard_size=4,
        row_group=2,
        buckets=2,
        zstd_level=3,
        max_members=None,
        skip_s1=False,
        connections=1,
        chunk_mb=1,
    )
    base.update(kw)
    return Namespace(**base)


def _read_all(mirror: Path) -> dict[str, dict]:
    rows = {}
    for f in sorted(mirror.glob("*/shard-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            assert r["patch_id"] not in rows, "image written twice"
            rows[r["patch_id"]] = r
    return rows


def _check_cubes(rows: dict[str, dict]) -> None:
    for pid, r in rows.items():
        cube = cube_from_bytes(r["bands"])
        raw = {b: synth.s2_band(pid, b) for b in S2_BANDS}
        assert np.array_equal(cube[: len(S2_BANDS)], encode_cube(raw, None, SPEC)[:10])
        vv = quantize_s1(synth.s1_band(r["s1_name"], "VV"), "VV", SPEC)
        assert r["has_s1"]
        assert np.array_equal(cube[CHANNELS.index("VV")], vv)


def test_build_end_to_end(ben_parquet: Path, ben_tars: tuple[Path, Path], tmp_path: Path) -> None:
    s2, s1 = ben_tars
    report = bs.build(_args(ben_parquet, str(s1), str(s2), tmp_path / "hub", tmp_path / "work"))
    rows = _read_all(tmp_path / "hub")
    assert len(rows) == 14
    assert report["s2"]["written"] == 14
    assert {r["split"] for r in rows.values()} == {"train", "val_small", "bench"}
    assert not any(r["split"] == "test" for r in rows.values())
    _check_cubes(rows)
    index = json.loads((tmp_path / "hub/meta/shard_index.json").read_text())
    assert sum(m["n_images"] for m in index["shards"].values()) == 14
    assert (tmp_path / "hub/meta/quantization.json").exists()
    assert "CDLA" in (tmp_path / "hub/README.md").read_text()
    totals = bs.index_totals(index)
    assert totals["train"]["images"] == 10
    assert totals["train"]["captioning:captioning"] == 10
    assert report["verify"].startswith("ok")
    sel = bs.select_images(ben_parquet, val_small=2, seed=0)
    index["shards"].popitem()
    assert bs.verify(index, sel)  # a missing shard is reported
    # local staging is cleaned once uploads land
    assert not list((tmp_path / "work/out").rglob("*.parquet"))


def test_build_resumes_without_duplicates(
    ben_parquet: Path, ben_tars: tuple[Path, Path], tmp_path: Path
) -> None:
    s2, s1 = ben_tars
    hub, work = tmp_path / "hub", tmp_path / "work"
    # first run: stop the S2 tar early (simulates a crash part-way); partial buckets flush
    bs.build(_args(ben_parquet, str(s1), str(s2), hub, work, max_members=12 * 7))
    first = _read_all(hub)
    assert 0 < len(first) < 14
    report = bs.build(_args(ben_parquet, str(s1), str(s2), hub, work))
    assert report["already_done"] == len(first)
    rows = _read_all(hub)  # asserts no image appears twice
    assert len(rows) == 14
    _check_cubes(rows)


def test_resumable_http_stream_survives_drops(tmp_path: Path) -> None:
    blob = np.random.default_rng(0).integers(0, 255, 300_000, dtype=np.uint8).tobytes()
    (tmp_path / "f.bin").write_bytes(blob)
    with synth.range_server({"f.bin": tmp_path / "f.bin"}, drops=[50_000, 1, 120_000]) as srv:
        stream = ResumableHTTPStream(synth.url_of(srv, "f.bin"), chunk_size=4096, backoff=0.01)
        got = b"".join(iter(lambda: stream.read(7777), b""))
        assert got == blob
        assert stream.reconnects == 3
        assert srv.requests == 4


def test_parallel_range_reader_orders_chunks_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("benvqa.data.httpstream.time.sleep", lambda s: None)
    blob = np.random.default_rng(1).integers(0, 255, 1_000_003, dtype=np.uint8).tobytes()
    (tmp_path / "f.bin").write_bytes(blob)
    # request 0 is the size probe; later drops hit chunk fetches (retried from their offset)
    with synth.range_server({"f.bin": tmp_path / "f.bin"}, drops=[10**9, 5_000, 7, 30_000]) as srv:
        reader = ParallelRangeReader(
            synth.url_of(srv, "f.bin"), connections=4, chunk_size=65_536, lookahead=3
        )
        got = b"".join(iter(lambda: reader.read(10_000), b""))
        reader.close()
    assert got == blob
    assert reader.retries == 3
    assert reader.n_chunks == 16


@pytest.mark.parametrize("connections", [1, 3])
def test_build_over_flaky_http(
    connections: int,
    ben_parquet: Path,
    ben_tars: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s2, s1 = ben_tars
    monkeypatch.setattr("benvqa.data.httpstream.time.sleep", lambda s: None)
    files = {"s1.tar.zst": s1, "s2.tar.zst": s2}
    with synth.range_server(files, drops=[30_000, 70_000, 5_000, 200_000]) as srv:
        bs.build(
            _args(
                ben_parquet,
                synth.url_of(srv, "s1.tar.zst"),
                synth.url_of(srv, "s2.tar.zst"),
                tmp_path / "hub",
                tmp_path / "work",
                connections=connections,
                chunk_mb=0.05,
            )
        )
        assert srv.requests >= 6
    rows = _read_all(tmp_path / "hub")
    assert len(rows) == 14
    _check_cubes(rows)


def test_local_mirror_listing(tmp_path: Path) -> None:
    m = LocalMirror(tmp_path / "m")
    (tmp_path / "a.txt").write_text("x")
    m.upload({"d/a.txt": tmp_path / "a.txt"}, "msg")
    assert m.list("d") == ["d/a.txt"]
    assert m.download("d/a.txt", tmp_path / "dl").read_text() == "x"
    with pytest.raises(FileNotFoundError):
        m.download("nope", tmp_path / "dl")
