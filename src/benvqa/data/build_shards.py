"""Stage 0: build compact per-image Parquet shards for BigEarthNet.txt (CPU only, resumable).

Pipeline (streaming mode, needs < 20 GB of disk):

1. Read ``BigEarthNet.txt.parquet`` and group rows per image (``patch_id``). Selection:
   all train images, ``val_small`` (N random validation images, fixed seed), all bench
   images, optionally the full validation split. Test images are never selected.
2. Pass 1: stream ``BigEarthNet-S1.tar.zst``; quantize VV/VH of needed patches into a
   disk-backed memmap ``[N, 2, 120, 120]`` uint8.
3. Pass 2: stream ``BigEarthNet-S2.tar.zst``; per needed patch build the 12-channel cube
   (10 S2 bands + the S1 codes) and hand it to a randomly chosen open shard "bucket".
   Tars are ordered by tile, so the buckets mix many tiles into every shard.
4. A full bucket becomes ``<split>/shard-XXXXX.parquet``. It is uploaded in one commit
   together with its patch list (``meta/patch_lists/...``) and the updated
   ``meta/shard_index.json``, then deleted locally.

Resume: the set of finished patches is exactly the union of the uploaded patch lists, so a
restart (even on a fresh machine) re-streams the tars but only decodes unfinished patches.
Connection drops inside a pass are absorbed by ``ResumableHTTPStream`` (HTTP Range).

    PYTHONPATH=src python -m benvqa.data.build_shards --parquet .../BigEarthNet.txt.parquet \
        --mirror hub:<user>/benvqa-shards --work-dir /kaggle/tmp/build
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import tarfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benvqa.data.httpstream import open_source
from benvqa.data.quantize import (
    CUBE_BYTES,
    S1_BANDS,
    S2_BANDS,
    SIZE,
    QuantSpec,
    encode_cube,
    quantize_s1,
)
from benvqa.hub_sync import AsyncUploader, Mirror, make_mirror

log = logging.getLogger("benvqa.build_shards")

ZENODO = "https://zenodo.org/api/records/10891137/files"
S1_URL = f"{ZENODO}/BigEarthNet-S1.tar.zst/content"
S2_URL = f"{ZENODO}/BigEarthNet-S2.tar.zst/content"
VQA_TYPES = ("binary", "mcq", "captioning")
NEVER = frozenset({"test"})

SCHEMA = pa.schema(
    [
        ("patch_id", pa.string()),
        ("s1_name", pa.string()),
        ("split", pa.string()),
        ("country", pa.string()),
        ("season", pa.string()),
        ("climate_zone", pa.string()),
        ("has_s1", pa.bool_()),
        ("bands", pa.binary(CUBE_BYTES)),
        ("qa", pa.string()),
        ("bbox", pa.string()),
    ],
    metadata={"bands": "uint8 [12,120,120] C-order; channel order in meta/quantization.json"},
)

LICENSE_README = """---
license: cdla-permissive-1.0
pretty_name: BigEarthNet.txt VQA shards (benvqa)
---
# benvqa shards

Per-image Parquet shards for training vision-language models on
[BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt).
One row = one image: a 12-channel uint8 cube (Sentinel-2 B02 B03 B04 B05 B06 B07 B08 B8A
B11 B12 + Sentinel-1 VV VH, 120x120 px at 10 m) plus all of its binary / MCQ / caption
annotations (JSON) and its bounding-box annotations (JSON, separate column).

Built by `benvqa.data.build_shards` from
* BigEarthNet.txt annotations (BIFOLD, CDLA-Permissive-1.0), and
* BigEarthNet v2.0 imagery, Zenodo record 10891137 (CDLA-Permissive-1.0).

Encoding parameters: `meta/quantization.json`. Shard statistics: `meta/shard_index.json`.
Test-split images are not included.

## Attribution
J. Herzog, M. Adler, L. Hackel, Y. Shu, A. Zavras, I. Papoutsis, P. Rota, B. Demir,
"BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth
Observation", arXiv:2603.29630, 2026.

K. N. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, V. Markl, "reBEN: Refined
BigEarthNet Dataset for Remote Sensing Image Analysis", IGARSS 2025.
"""


# --------------------------------------------------------------------------- selection
@dataclass
class ImageRecord:
    patch_id: str
    s1_name: str
    split: str
    """Shard split: train / val_small / val / bench."""
    country: str
    season: str
    climate_zone: str
    qa: str
    bbox: str
    counts: dict[str, int]


@dataclass
class Selection:
    images: dict[str, ImageRecord]
    s1_index: dict[str, int]
    """s1_name -> row in the S1 memmap."""

    def by_split(self) -> Counter[str]:
        return Counter(r.split for r in self.images.values())


def _hash01(*parts: object) -> float:
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2**64


def select_images(
    parquet: Path,
    *,
    val_small: int | str = 3000,
    include_full_val: bool = False,
    seed: int = 1234,
) -> Selection:
    cols = [
        "ID",
        "patch_id",
        "s1_name",
        "input",
        "output",
        "type",
        "category",
        "split",
        "country",
        "season",
        "climate_zone",
    ]
    df = pq.read_table(parquet, columns=cols).to_pandas()
    df = df[~df["split"].isin(NEVER)]
    df.loc[df["type"] == "captioning", "category"] = "captioning"

    split_of_patch = df.groupby("patch_id")["split"].agg(lambda s: set(s))
    multi = split_of_patch[split_of_patch.map(len) > 1]
    assert multi.empty, f"images in several splits: {list(multi.index[:5])}"
    split_of_patch = split_of_patch.map(lambda s: next(iter(s)))

    val_ids = sorted(split_of_patch.index[split_of_patch == "validation"])
    if val_small == "all":
        chosen_val = set(val_ids)
    else:
        rng = np.random.default_rng(seed)
        chosen_val = set(rng.choice(val_ids, size=min(int(val_small), len(val_ids)), replace=False))
    shard_split = {}
    for pid, split in split_of_patch.items():
        if split == "validation":
            if pid in chosen_val:
                shard_split[pid] = "val_small"
            elif include_full_val:
                shard_split[pid] = "val"
        else:
            shard_split[pid] = split  # train / bench
    train = {p for p, s in shard_split.items() if s == "train"}
    held_out = {p for p, s in shard_split.items() if s in ("bench", "val_small", "val")}
    assert not train & held_out, "train overlaps held-out images"

    df = df[df["patch_id"].isin(shard_split)].sort_values(["patch_id", "ID"], kind="stable")
    images: dict[str, ImageRecord] = {}
    pids = df["patch_id"].to_numpy()
    bounds = np.flatnonzero(pids[1:] != pids[:-1]) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(pids)]
    cols_np = {c: df[c].to_numpy() for c in cols}
    for s, e in zip(starts, ends, strict=True):
        pid = str(cols_np["patch_id"][s])
        qa, bbox, counts = [], [], Counter()
        for i in range(s, e):
            item = {
                "id": int(cols_np["ID"][i]),
                "type": str(cols_np["type"][i]),
                "category": str(cols_np["category"][i]),
                "input": str(cols_np["input"][i]),
                "output": str(cols_np["output"][i]),
            }
            counts[f"{item['type']}:{item['category']}"] += 1
            (qa if item["type"] in VQA_TYPES else bbox).append(item)
        images[pid] = ImageRecord(
            patch_id=pid,
            s1_name=str(cols_np["s1_name"][s]),
            split=shard_split[pid],
            country=str(cols_np["country"][s]),
            season=str(cols_np["season"][s]),
            climate_zone=str(cols_np["climate_zone"][s]),
            qa=json.dumps(qa, ensure_ascii=False, separators=(",", ":")),
            bbox=json.dumps(bbox, ensure_ascii=False, separators=(",", ":")),
            counts=dict(counts),
        )
    s1_names = sorted({r.s1_name for r in images.values()})
    return Selection(images, {n: i for i, n in enumerate(s1_names)})


# ------------------------------------------------------------------------------ tar io
def iter_tar(
    src: IO[bytes], max_members: int | None = None
) -> Iterator[tuple[tarfile.TarFile, tarfile.TarInfo]]:
    """Regular-file members of a ``.tar.zst`` stream. Unread members are skipped cheaply."""
    import zstandard

    reader = zstandard.ZstdDecompressor().stream_reader(src, read_size=1 << 20)
    with tarfile.open(fileobj=reader, mode="r|") as tf:
        for i, member in enumerate(tf):
            if max_members is not None and i >= max_members:
                return
            if member.isfile():
                yield tf, member


def read_tif(data: bytes) -> np.ndarray:
    import tifffile

    return tifffile.imread(io.BytesIO(data))


def split_member(name: str) -> tuple[str, str]:
    """``.../<patch>/<patch>_<band>.tif`` -> (patch, band)."""
    parts = name.rsplit("/", 2)
    stem = parts[-1].removesuffix(".tif")
    patch = parts[-2] if len(parts) >= 2 else stem.rsplit("_", 1)[0]
    return patch, stem.rsplit("_", 1)[-1]


# ----------------------------------------------------------------------------- S1 pass
class S1Store:
    """Disk-backed ``[N, 2, 120, 120]`` uint8 S1 codes with a persisted fill bitmap."""

    def __init__(self, work_dir: Path, n: int) -> None:
        self.path = work_dir / "s1.u8"
        self.filled_path = work_dir / "s1_filled.npy"
        mode = "r+" if self.path.exists() else "w+"
        self.data = np.memmap(self.path, dtype=np.uint8, mode=mode, shape=(n, 2, SIZE, SIZE))
        self.filled = (
            np.load(self.filled_path)
            if self.filled_path.exists() and mode == "r+"
            else np.zeros((n, 2), dtype=bool)
        )
        assert self.filled.shape == (n, 2), "S1 store does not match the selection"

    def flush(self) -> None:
        self.data.flush()
        tmp = self.filled_path.with_suffix(".tmp.npy")
        np.save(tmp, self.filled)
        tmp.replace(self.filled_path)


def s1_pass(
    url: str,
    sel: Selection,
    store: S1Store,
    spec: QuantSpec,
    needed: set[str],
    max_members: int | None = None,
    src_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0, stats = time.time(), Counter()
    band_idx = {b: i for i, b in enumerate(S1_BANDS)}
    with open_source(url, **(src_kwargs or {})) as src:
        for k, (tf, m) in enumerate(iter_tar(src, max_members)):
            stats["members"] += 1
            s1_name, band = split_member(m.name)
            row = sel.s1_index.get(s1_name)
            if row is None or s1_name not in needed or band not in band_idx:
                continue
            j = band_idx[band]
            if store.filled[row, j]:
                continue
            fh = tf.extractfile(m)
            assert fh is not None
            arr = read_tif(fh.read())
            if arr.shape != (SIZE, SIZE):
                stats["bad_shape"] += 1
                continue
            store.data[row, j] = quantize_s1(arr, band, spec)
            store.filled[row, j] = True
            stats["decoded"] += 1
            if k % 200_000 == 0:
                store.flush()
                log.info("S1: %d members, %d decoded, %.0fs", k, stats["decoded"], time.time() - t0)
    store.flush()
    stats["seconds"] = round(time.time() - t0)
    return dict(stats)


# ------------------------------------------------------------------------------ shards
@dataclass
class Bucket:
    rows: list[dict[str, Any]] = field(default_factory=list)
    """Rows not yet written as a row group."""
    writer: pq.ParquetWriter | None = None
    path: Path | None = None
    patch_ids: list[str] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


class ShardBuilder:
    def __init__(
        self,
        out_dir: Path,
        mirror: Mirror,
        uploader: AsyncUploader,
        index: dict[str, Any],
        *,
        shard_size: int = 1000,
        row_group: int = 50,
        n_buckets: dict[str, int],
        zstd_level: int = 6,
        seed: int = 1234,
    ) -> None:
        self.out_dir, self.mirror, self.uploader = out_dir, mirror, uploader
        self.index = index
        self.shard_size, self.row_group, self.zstd_level = shard_size, row_group, zstd_level
        self.n_buckets, self.seed = n_buckets, seed
        self.buckets: dict[tuple[str, int], Bucket] = defaultdict(Bucket)
        self._tmp_seq = 0

    def add(self, rec: ImageRecord, cube: np.ndarray, has_s1: bool) -> None:
        n = self.n_buckets[rec.split]
        b_id = int(_hash01(self.seed, "bucket", rec.patch_id) * n)
        bucket = self.buckets[(rec.split, b_id)]
        bucket.rows.append(
            {
                "patch_id": rec.patch_id,
                "s1_name": rec.s1_name,
                "split": rec.split,
                "country": rec.country,
                "season": rec.season,
                "climate_zone": rec.climate_zone,
                "has_s1": has_s1,
                "bands": cube.tobytes(),
                "qa": rec.qa,
                "bbox": rec.bbox,
            }
        )
        bucket.patch_ids.append(rec.patch_id)
        bucket.counts.update(rec.counts)
        bucket.counts["images"] += 1
        if len(bucket.rows) >= self.row_group:
            self._write_rows(rec.split, bucket)
        if len(bucket.patch_ids) >= self.shard_size:
            self._close(rec.split, b_id)

    def _write_rows(self, split: str, bucket: Bucket) -> None:
        if not bucket.rows:
            return
        if bucket.writer is None:
            self._tmp_seq += 1
            bucket.path = self.out_dir / "tmp" / f"{split}-{self._tmp_seq:06d}.parquet"
            bucket.path.parent.mkdir(parents=True, exist_ok=True)
            bucket.writer = pq.ParquetWriter(
                bucket.path,
                SCHEMA,
                compression="zstd",
                compression_level=self.zstd_level,
                use_dictionary=["split", "country", "season", "climate_zone"],
                write_statistics=["patch_id"],
            )
        table = pa.Table.from_pylist(bucket.rows, schema=SCHEMA)
        bucket.writer.write_table(table, row_group_size=self.row_group)
        bucket.rows = []

    def _close(self, split: str, b_id: int) -> None:
        bucket = self.buckets.pop((split, b_id))
        self._write_rows(split, bucket)
        if bucket.writer is None or bucket.path is None:
            return
        bucket.writer.close()
        num = self.index["next_shard"].get(split, 0)
        self.index["next_shard"][split] = num + 1
        rel = f"{split}/shard-{num:05d}.parquet"
        final = self.out_dir / rel
        final.parent.mkdir(parents=True, exist_ok=True)
        bucket.path.replace(final)
        plist = self.out_dir / f"meta/patch_lists/{split}/shard-{num:05d}.txt"
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_text("\n".join(bucket.patch_ids) + "\n", encoding="utf-8")
        counts = dict(bucket.counts)
        self.index["shards"][rel] = {
            "split": split,
            "n_images": counts.pop("images"),
            "bytes": final.stat().st_size,
            "n_rows": dict(sorted(counts.items())),
        }
        idx_path = self.out_dir / f"meta/shard_index.{num}.{split}.json"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(self.index, indent=1) + "\n", encoding="utf-8")
        files = {
            rel: final,
            f"meta/patch_lists/{split}/shard-{num:05d}.txt": plist,
            "meta/shard_index.json": idx_path,
        }

        def cleanup() -> None:
            for p in (final, plist, idx_path):
                p.unlink(missing_ok=True)

        log.info(
            "shard %s: %d images, %.1f MB", rel, len(bucket.patch_ids), final.stat().st_size / 1e6
        )
        self.uploader.submit(files, f"add {rel}", on_done=cleanup)

    def flush_all(self) -> None:
        for split, b_id in sorted(self.buckets):
            self._close(split, b_id)


def load_progress(mirror: Mirror, work_dir: Path) -> tuple[dict[str, Any], set[str]]:
    """Remote shard index + union of uploaded patch lists (shards missing from the repo are
    dropped from the index, so their patches are rebuilt)."""
    index: dict[str, Any] = {"version": 1, "next_shard": {}, "shards": {}}
    try:
        idx_file = mirror.download("meta/shard_index.json", work_dir / "remote")
        index = json.loads(idx_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return index, set()
    present = set(mirror.list(""))
    done: set[str] = set()
    for rel in list(index["shards"]):
        split, name = rel.split("/")
        plist = f"meta/patch_lists/{split}/{name.replace('.parquet', '.txt')}"
        if rel not in present or plist not in present:
            log.warning("index lists %s but it is missing on the mirror; rebuilding it", rel)
            del index["shards"][rel]
            continue
        text = mirror.download(plist, work_dir / "remote").read_text(encoding="utf-8")
        done.update(text.split())
    return index, done


# ----------------------------------------------------------------------------- S2 pass
def s2_pass(
    url: str,
    sel: Selection,
    store: S1Store | None,
    spec: QuantSpec,
    builder: ShardBuilder,
    todo: set[str],
    uploader: AsyncUploader,
    max_members: int | None = None,
    src_kwargs: dict[str, Any] | None = None,
    require_s1: bool = False,
) -> dict[str, Any]:
    t0, stats = time.time(), Counter()
    current: str | None = None
    bands: dict[str, np.ndarray] = {}
    finished: set[str] = set()

    def finish(pid: str) -> None:
        rec = sel.images[pid]
        missing = [b for b in S2_BANDS if b not in bands]
        if missing:
            stats["incomplete_s2"] += 1
            log.warning("patch %s is missing bands %s; skipped", pid, missing)
            return
        row = sel.s1_index[rec.s1_name]
        has_s1 = store is not None and bool(store.filled[row].all())
        if require_s1 and not has_s1:
            stats["skipped_no_s1"] += 1
            return
        s1 = np.asarray(store.data[row]) if has_s1 and store is not None else None
        stats["with_s1" if has_s1 else "without_s1"] += 1
        builder.add(rec, encode_cube(bands, s1, spec), has_s1)
        stats["written"] += 1
        finished.add(pid)

    with open_source(url, **(src_kwargs or {})) as src:
        for k, (tf, m) in enumerate(iter_tar(src, max_members)):
            stats["members"] += 1
            pid, band = split_member(m.name)
            if pid != current:
                if current is not None and current in todo and current not in finished:
                    finish(current)
                current, bands = pid, {}
                if pid in finished:
                    stats["patch_reappeared"] += 1
            if pid not in todo or pid in finished or band not in S2_BANDS:
                continue
            fh = tf.extractfile(m)
            assert fh is not None
            bands[band] = read_tif(fh.read())
            if k % 100_000 == 0:
                if uploader.errors:
                    raise RuntimeError(f"upload failed: {uploader.errors[-1]}")
                log.info("S2: %d members, %d written, %.0fs", k, stats["written"], time.time() - t0)
        if current is not None and current in todo and current not in finished:
            finish(current)
    stats["seconds"] = round(time.time() - t0)
    return dict(stats)


# -------------------------------------------------------------------------------- main
def default_buckets(by_split: Counter[str], shard_size: int, max_buckets: int) -> dict[str, int]:
    return {s: max(1, min(max_buckets, n // shard_size)) for s, n in by_split.items()}


def build(args: argparse.Namespace) -> dict[str, Any]:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    spec = QuantSpec()
    sel = select_images(
        args.parquet,
        val_small=args.val_small,
        include_full_val=args.include_full_val,
        seed=args.seed,
    )
    log.info("selected %s", dict(sel.by_split()))
    mirror = make_mirror(
        args.mirror,
        **(
            {"create": True, "private": args.private}
            if not args.mirror.startswith("local:")
            else {}
        ),
    )
    index, done = load_progress(mirror, work)
    todo = set(sel.images) - done
    log.info("%d images already uploaded, %d to do", len(done), len(todo))

    uploader = AsyncUploader(mirror, "shard-uploader")
    meta = work / "meta_out"
    meta.mkdir(exist_ok=True)
    (meta / "quantization.json").write_text(spec.to_json(), encoding="utf-8")
    (meta / "README.md").write_text(LICENSE_README, encoding="utf-8")
    uploader.submit(
        {"meta/quantization.json": meta / "quantization.json", "README.md": meta / "README.md"},
        "metadata",
    )

    report: dict[str, Any] = {"selected": dict(sel.by_split()), "already_done": len(done)}
    src_kwargs: dict[str, Any] = {"connections": args.connections}
    if args.connections > 1:
        src_kwargs["chunk_size"] = int(args.chunk_mb * (1 << 20))
    store = None
    if not args.skip_s1:
        store = S1Store(work, len(sel.s1_index))
        needed = {sel.images[p].s1_name for p in todo}
        max_s1 = getattr(args, "max_members_s1", None) or args.max_members
        report["s1"] = s1_pass(args.s1_url, sel, store, spec, needed, max_s1, src_kwargs)
        log.info("S1 pass: %s", report["s1"])
    builder = ShardBuilder(
        work / "out",
        mirror,
        uploader,
        index,
        shard_size=args.shard_size,
        row_group=args.row_group,
        zstd_level=args.zstd_level,
        n_buckets=default_buckets(sel.by_split(), args.shard_size, args.buckets),
        seed=args.seed,
    )
    max_s2 = getattr(args, "max_members_s2", None) or args.max_members
    report["s2"] = s2_pass(
        args.s2_url,
        sel,
        store,
        spec,
        builder,
        todo,
        uploader,
        max_s2,
        src_kwargs,
        require_s1=getattr(args, "require_s1", False),
    )
    builder.flush_all()
    uploader.close()
    if uploader.errors:
        raise RuntimeError(f"uploads failed: {uploader.errors}")
    report["shards"] = len(index["shards"])
    report["totals"] = index_totals(index)
    if args.max_members is None and not getattr(args, "max_members_s2", None):
        report["verify"] = verify(index, sel) or "ok: index matches the parquet selection"
    log.info("done: %s", json.dumps(report, indent=1))
    return report


def expected_totals(sel: Selection) -> dict[str, dict[str, int]]:
    tot: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in sel.images.values():
        tot[rec.split]["images"] += 1
        tot[rec.split].update(rec.counts)
    return {k: dict(v) for k, v in tot.items()}


def verify(index: dict[str, Any], sel: Selection) -> list[str]:
    """Differences between the shard index and the parquet selection (empty = complete)."""
    got, want = index_totals(index), expected_totals(sel)
    problems = []
    for split, counts in want.items():
        for key, n in counts.items():
            if got.get(split, {}).get(key, 0) != n:
                problems.append(f"{split}/{key}: index {got.get(split, {}).get(key, 0)} != {n}")
    problems += [f"unexpected split {s} in index" for s in set(got) - set(want)]
    return problems


def index_totals(index: dict[str, Any]) -> dict[str, Any]:
    tot: dict[str, Counter[str]] = defaultdict(Counter)
    for meta in index["shards"].values():
        tot[meta["split"]]["images"] += meta["n_images"]
        tot[meta["split"]]["bytes"] += meta["bytes"]
        tot[meta["split"]].update(meta["n_rows"])
    return {k: dict(v) for k, v in tot.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--mirror", required=True, help="hub:<user>/benvqa-shards or local:<dir>")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    ap.add_argument("--work-dir", type=Path, required=True, help="S1 memmap + staging (~10 GB)")
    ap.add_argument("--s1-url", default=S1_URL)
    ap.add_argument("--s2-url", default=S2_URL)
    ap.add_argument("--val-small", default="3000", help="N validation images, or 'all'")
    ap.add_argument("--include-full-val", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--row-group", type=int, default=50)
    ap.add_argument("--buckets", type=int, default=32)
    ap.add_argument("--zstd-level", type=int, default=9, help="9: ~130 KB/image, 310 img/s")
    ap.add_argument(
        "--connections",
        type=int,
        default=16,
        help="parallel HTTP range connections (Zenodo caps ~0.4 MB/s each)",
    )
    ap.add_argument("--chunk-mb", type=float, default=32, help="range request size")
    ap.add_argument("--max-members", type=int, default=None, help="dry run: stop each tar early")
    ap.add_argument("--max-members-s1", type=int, default=None, help="overrides --max-members")
    ap.add_argument("--max-members-s2", type=int, default=None, help="overrides --max-members")
    ap.add_argument(
        "--require-s1",
        action="store_true",
        help="dev builds: skip patches whose S1 was not seen (partial S1 pass)",
    )
    ap.add_argument("--skip-s1", action="store_true", help="no S1 pass (VV/VH left at zero)")
    args = ap.parse_args(argv)
    if args.val_small != "all":
        args.val_small = int(args.val_small)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per range request otherwise
    report = build(args)
    (Path(args.work_dir) / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
