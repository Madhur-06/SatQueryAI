"""Stage 2: deterministic, resumable micro-batch streams from shards on the Hub.

Layout of the sample stream (everything is a pure function of the config seed):

* There are ``G = world_size x num_workers`` streams. For epoch ``e``, the train shards are
  permuted with ``(seed, e)`` and stream ``s`` takes ``perm[s::G]``, read in groups of
  ``interleave`` shards whose rows are interleaved round-robin (each shard in its own
  deterministic row order).
* Consecutive rows form **pools** of ``pool_size`` conversations. A pool never crosses a
  group boundary. Inside a pool, conversations are sorted by token length and cut into
  micro-batches of ``micro_batch`` (little padding). The micro-batch order is then shuffled.
* The **cursor** of a stream is the number of micro-batches the trainer has consumed. Pools
  and groups hold a known number of micro-batches (``ceil(n / micro_batch)``, from
  ``meta/shard_index.json``), so resuming skips whole groups and pools arithmetically with
  no download. Only the current pool's text is rebuilt (no images).
* DataLoader workers deliver round-robin. Worker ``w`` of rank ``r`` serves stream
  ``r * num_workers + (w + rotation) % num_workers``, with ``rotation`` = micro-batches
  consumed by the rank modulo ``num_workers``. A resumed run therefore continues exactly
  where the old one would have.

Each stream prefetches its next shards on a background thread into its own cache
directory, and deletes them once consumed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch.utils.data as tud

from benvqa.config import Config
from benvqa.data.collate import Item, collate
from benvqa.data.composites import make_composite, resize, resolve_composites
from benvqa.data.conversation import build_conversation, parse_qa, rng_for
from benvqa.data.quantize import CUBE_SHAPE, QuantSpec
from benvqa.data.render import Renderer
from benvqa.data.vision import patchify_u8
from benvqa.hub_sync import Mirror, make_mirror

log = logging.getLogger(__name__)
COLUMNS = ["patch_id", "split", "has_s1", "bands", "qa"]


@dataclass(frozen=True)
class ShardInfo:
    path: str
    n_images: int


@dataclass(frozen=True)
class Position:
    epoch: int
    group: int
    pool: int
    batch: int
    """micro-batch index inside the pool"""


def load_shards(mirror: Mirror, cache_dir: Path, split: str) -> list[ShardInfo]:
    index = json.loads(mirror.download("meta/shard_index.json", cache_dir / "meta").read_text())
    shards = [
        ShardInfo(p, m["n_images"]) for p, m in index["shards"].items() if m["split"] == split
    ]
    if not shards:
        raise FileNotFoundError(f"no {split!r} shards in {mirror.describe()}")
    return sorted(shards, key=lambda s: s.path)


def load_quant_spec(mirror: Mirror, cache_dir: Path) -> QuantSpec:
    text = mirror.download("meta/quantization.json", cache_dir / "meta").read_text()
    return QuantSpec.from_json(text)


class StreamPlan:
    def __init__(
        self,
        shards: list[ShardInfo],
        seed: int,
        n_streams: int,
        interleave: int,
        pool_size: int,
        micro_batch: int,
    ) -> None:
        if len(shards) < n_streams:
            raise ValueError(f"{len(shards)} shards < {n_streams} streams")
        if pool_size % micro_batch:
            raise ValueError("pool_size must be a multiple of micro_batch")
        self.shards, self.seed, self.n_streams = shards, seed, n_streams
        self.interleave, self.pool_size, self.micro_batch = interleave, pool_size, micro_batch

    def groups(self, stream: int, epoch: int) -> list[tuple[ShardInfo, ...]]:
        perm = np.random.default_rng([self.seed, 1, epoch]).permutation(len(self.shards))
        mine = [self.shards[i] for i in perm[stream :: self.n_streams]]
        k = self.interleave
        return [tuple(mine[i : i + k]) for i in range(0, len(mine), k)]

    def row_order(self, group: tuple[ShardInfo, ...], epoch: int) -> list[tuple[int, int]]:
        orders = [
            rng_for(self.seed, s.path, epoch, "rows").permutation(s.n_images).tolist()
            for s in group
        ]
        out: list[tuple[int, int]] = []
        for i in range(max(len(o) for o in orders)):
            out += [(j, o[i]) for j, o in enumerate(orders) if i < len(o)]
        return out

    def pool_sizes(self, n_rows: int) -> list[int]:
        full, rest = divmod(n_rows, self.pool_size)
        return [self.pool_size] * full + ([rest] if rest else [])

    def n_batches(self, pool_size: int) -> int:
        return math.ceil(pool_size / self.micro_batch)

    def group_batches(self, group: tuple[ShardInfo, ...]) -> int:
        return sum(self.n_batches(p) for p in self.pool_sizes(sum(s.n_images for s in group)))

    def locate(self, stream: int, cursor: int) -> Position:
        """Arithmetic skip of ``cursor`` micro-batches (no data access)."""
        epoch = 0
        while True:
            groups = self.groups(stream, epoch)
            per_epoch = sum(self.group_batches(g) for g in groups)
            if cursor < per_epoch:
                break
            cursor -= per_epoch
            epoch += 1
        for gi, g in enumerate(groups):
            nb = self.group_batches(g)
            if cursor < nb:
                for pi, size in enumerate(self.pool_sizes(sum(s.n_images for s in g))):
                    if cursor < self.n_batches(size):
                        return Position(epoch, gi, pi, cursor)
                    cursor -= self.n_batches(size)
            cursor -= nb
        raise AssertionError("unreachable")

    def batch_order(self, lengths: list[int], key: tuple[int, ...]) -> list[list[int]]:
        idx = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
        chunks = [idx[i : i + self.micro_batch] for i in range(0, len(idx), self.micro_batch)]
        perm = np.random.default_rng([self.seed, 2, *key]).permutation(len(chunks))
        return [chunks[i] for i in perm]

    def upcoming_shards(self, stream: int, pos: Position) -> Iterator[ShardInfo]:
        epoch, gi = pos.epoch, pos.group
        while True:
            groups = self.groups(stream, epoch)
            for g in groups[gi:]:
                yield from g
            epoch, gi = epoch + 1, 0


class ShardFetcher:
    """Background downloads of a stream's upcoming shards (bounded look-ahead)."""

    def __init__(
        self,
        mirror: Mirror,
        cache_dir: Path,
        upcoming: Iterator[ShardInfo],
        lookahead: int,
        budget_bytes: float,
    ) -> None:
        self.mirror, self.cache_dir = mirror, cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._upcoming, self.lookahead, self.budget = upcoming, lookahead, budget_bytes
        self._ready: dict[str, Path] = {}
        self._refs: dict[str, int] = {}
        """Pending uses per shard (a shard can recur across an epoch boundary)."""
        self._queue: list[str] = []
        self._cond = threading.Condition()
        self._error: BaseException | None = None
        self._stop = False
        self.download_seconds = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="shard-fetch")
        self._thread.start()

    def _usage(self) -> int:
        return sum(p.stat().st_size for p in self._ready.values() if p.exists())

    def _run(self) -> None:
        try:
            for shard in self._upcoming:
                with self._cond:
                    while not self._stop and (
                        len(self._queue) >= self.lookahead or self._usage() > self.budget
                    ):
                        self._cond.wait(timeout=1.0)
                    if self._stop:
                        return
                    self._queue.append(shard.path)
                    self._refs[shard.path] = self._refs.get(shard.path, 0) + 1
                    if shard.path in self._ready:
                        continue
                t0 = time.perf_counter()
                path = self.mirror.download(shard.path, self.cache_dir)
                self.download_seconds += time.perf_counter() - t0
                with self._cond:
                    self._ready[shard.path] = path
                    self._cond.notify_all()
        except BaseException as exc:  # surfaced to the consumer in get()
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    def get(self, rel: str) -> Path:
        with self._cond:
            while rel not in self._ready:
                if self._error is not None:
                    raise RuntimeError(f"shard download failed: {rel}") from self._error
                self._cond.wait(timeout=1.0)
            return self._ready[rel]

    def release(self, rel: str) -> None:
        path = None
        with self._cond:
            if rel in self._queue:
                self._queue.remove(rel)
            self._refs[rel] = self._refs.get(rel, 1) - 1
            if self._refs[rel] <= 0:
                self._refs.pop(rel, None)
                path = self._ready.pop(rel, None)
            self._cond.notify_all()
        if path is not None:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()


def bands_array(table: pa.Table) -> np.ndarray:
    """Zero-copy ``[n, 12, 120, 120]`` view of the fixed-size ``bands`` column."""
    col = table.column("bands").combine_chunks()
    buf = col.buffers()[1]
    n = len(col)
    start = col.offset * int(np.prod(CUBE_SHAPE))
    flat = np.frombuffer(buf, dtype=np.uint8, count=n * int(np.prod(CUBE_SHAPE)), offset=start)
    return flat.reshape(n, *CUBE_SHAPE)


class ImageMaker:
    def __init__(self, cfg: Config, spec: QuantSpec) -> None:
        self.size = cfg.images.size
        self.comps = resolve_composites(cfg.images.composites, cfg.images.composite_params)
        self.spec = spec

    def tiles(self, cube: np.ndarray, names: tuple[str, ...]) -> tuple[list, list]:
        tiles, grids = [], []
        for name in names:
            img = resize(make_composite(cube, self.comps[name], self.spec), self.size)
            t, g = patchify_u8(img)
            tiles.append(t)
            grids.append(g)
        return tiles, grids


class TrainStream(tud.IterableDataset):
    """Iterable of collated micro-batches for one rank (with or without DataLoader workers)."""

    def __init__(
        self,
        cfg: Config,
        rank: int,
        world_size: int,
        cursors: dict[int, int] | None = None,
        rotation: int = 0,
        tokenizer: Any = None,
        mirror_spec: str | None = None,
    ) -> None:
        if cfg.data.split != "train":
            raise ValueError("the training stream reads the train split only")
        self.cfg, self.rank, self.world_size = cfg, rank, world_size
        self.cursors = dict(cursors or {})
        self.rotation = rotation
        self.tokenizer = tokenizer
        self.mirror_spec = mirror_spec or cfg.data.hub_dataset
        self.n_workers = max(1, cfg.data.num_workers)
        self.n_streams = world_size * self.n_workers

    # -- per-worker setup (runs inside the worker process) --------------------------------
    def _setup(self, worker_id: int) -> None:
        cfg = self.cfg
        kw = (
            {}
            if self.mirror_spec.startswith(("local:", "flaky:"))
            else {"revision": cfg.data.revision}
        )
        self.mirror = make_mirror(self.mirror_spec, **kw)
        self.cache = Path(cfg.data.cache_dir)
        meta_dir = self.cache / f"meta-r{self.rank}-w{worker_id}"
        shards = load_shards(self.mirror, meta_dir, "train")
        assert all(s.path.startswith("train/") for s in shards)
        self.plan = StreamPlan(
            shards,
            cfg.seed,
            self.n_streams,
            cfg.data.interleave,
            cfg.train.pool_size,
            cfg.train.micro_batch,
        )
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.tokenizer, local_files_only=cfg.model.local_files_only
            )
        self.renderer = Renderer(self.tokenizer, cfg.images.size, cfg.train.max_len)
        self.images = ImageMaker(cfg, load_quant_spec(self.mirror, meta_dir))
        self.conv_cfg = cfg.conversation_config()

    def stream_id(self, worker_id: int) -> int:
        return self.rank * self.n_workers + (worker_id + self.rotation) % self.n_workers

    def __iter__(self) -> Iterator[dict]:
        info = tud.get_worker_info()
        worker_id = info.id if info is not None else 0
        if info is not None and info.num_workers != self.n_workers:
            raise ValueError("DataLoader num_workers differs from data.num_workers")
        self._setup(worker_id)
        sid = self.stream_id(worker_id)
        if info is None and self.n_workers > 1:  # in-process: interleave all local streams
            yield from self._round_robin()
            return
        yield from self.iter_stream(sid, self.cursors.get(sid, 0))

    def _round_robin(self) -> Iterator[dict]:
        its = [
            self.iter_stream(self.stream_id(w), self.cursors.get(self.stream_id(w), 0))
            for w in range(self.n_workers)
        ]
        while True:
            for it in its:
                yield next(it)

    def iter_stream(self, sid: int, start: int) -> Iterator[dict]:
        cfg, plan = self.cfg, self.plan
        pos = plan.locate(sid, start)
        budget = cfg.data.cache_budget_gb * 1e9 / self.n_streams
        fetcher = ShardFetcher(
            self.mirror,
            self.cache / f"stream-{sid:03d}",
            plan.upcoming_shards(sid, pos),
            lookahead=cfg.data.interleave + cfg.data.prefetch_shards,
            budget_bytes=budget,
        )
        counter = start
        try:
            epoch, gi, pi, bi = pos.epoch, pos.group, pos.pool, pos.batch
            while True:
                groups = plan.groups(sid, epoch)
                while gi < len(groups):
                    group = groups[gi]
                    tables = [self._read(fetcher.get(s.path)) for s in group]
                    order = plan.row_order(group, epoch)
                    pools = plan.pool_sizes(len(order))
                    offset = sum(pools[:pi])
                    for p in range(pi, len(pools)):
                        rows = order[offset : offset + pools[p]]
                        offset += pools[p]
                        convs = [self._conversation(tables[j], r, epoch) for j, r in rows]
                        rendered = [self.renderer.render(c) for c, _ in convs]
                        batches = plan.batch_order([len(r) for r in rendered], (epoch, sid, gi, p))
                        for b in range(bi, len(batches)):
                            items = [self._item(convs[i], rendered[i]) for i in batches[b]]
                            meta = {
                                "stream_id": sid,
                                "batch_index": counter,
                                "epoch": epoch,
                                "rank": self.rank,
                            }
                            yield collate(
                                items,
                                self.renderer.pad_id,
                                self.renderer.image_pad,
                                meta,
                                cfg.train.pad_to_multiple,
                            )
                            counter += 1
                        bi = 0
                    for s in group:
                        fetcher.release(s.path)
                    gi, pi = gi + 1, 0
                epoch, gi = epoch + 1, 0
        finally:
            fetcher.close()

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        table = pq.read_table(path, columns=COLUMNS)
        splits = set(table.column("split").to_pylist())
        if splits != {"train"}:
            raise AssertionError(f"{path} contains non-train rows: {splits}")
        return {
            "bands": bands_array(table),
            "qa": table.column("qa").to_pylist(),
            "patch_id": table.column("patch_id").to_pylist(),
        }

    def _conversation(self, table: dict[str, Any], row: int, epoch: int) -> tuple[Any, np.ndarray]:
        pid = table["patch_id"][row]
        conv = build_conversation(
            parse_qa(table["qa"][row]), pid, epoch, self.conv_cfg, self.cfg.seed
        )
        return conv, table["bands"][row]

    def _item(self, conv_cube: tuple[Any, np.ndarray], rendered: Any) -> Item:
        conv, cube = conv_cube
        tiles, grids = self.images.tiles(cube, conv.composites)
        return Item(rendered, tiles, grids, conv.patch_id)


def make_train_loader(stream: TrainStream, pin_memory: bool = True) -> tud.DataLoader:
    nw = stream.cfg.data.num_workers
    kwargs: dict[str, Any] = {}
    if nw > 0:
        kwargs = {"prefetch_factor": stream.cfg.data.prefetch_factor, "persistent_workers": True}
    return tud.DataLoader(stream, batch_size=None, num_workers=nw, pin_memory=pin_memory, **kwargs)


def default_cache_dir(preferred: str) -> str:
    """``/kaggle/tmp/...`` when it exists, else the system temp dir."""
    root = Path(preferred).parent
    if root.exists() and os.access(root, os.W_OK):
        return preferred
    import tempfile

    return str(Path(tempfile.gettempdir()) / "benvqa_shards")
