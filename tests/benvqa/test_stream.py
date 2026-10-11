"""Stage 2: shard plan arithmetic, exact resume, worker order, flaky network, split guard."""

from __future__ import annotations

import shutil
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from benvqa import testing as synth  # noqa: E402
from benvqa.config import Config, load_config  # noqa: E402
from benvqa.data import build_shards as bs  # noqa: E402
from benvqa.data.stream import (  # noqa: E402
    ShardInfo,
    StreamPlan,
    TrainStream,
    make_train_loader,
)


@pytest.fixture(scope="module")
def hub(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stream")
    images = synth.make_images({"train": 44, "validation": 4, "bench": 2}, seed=3)
    parquet = synth.write_parquet(root / "ann.parquet", images, seed=3)
    s2, s1 = synth.write_tars(root / "tars", images, extra=0)
    args = Namespace(
        parquet=parquet,
        mirror=f"local:{root / 'hub'}",
        private=False,
        work_dir=root / "work",
        s1_url=str(s1),
        s2_url=str(s2),
        val_small="all",
        include_full_val=False,
        seed=0,
        shard_size=4,
        row_group=2,
        buckets=3,
        zstd_level=1,
        max_members=None,
        skip_s1=False,
        connections=1,
        chunk_mb=1,
    )
    bs.build(args)
    return root / "hub"


def _cfg(hub: Path, tmp_path: Path, **over: object) -> Config:
    items = [
        f"data.hub_dataset=local:{hub}",
        f"data.cache_dir={tmp_path / 'cache'}",
        "data.num_workers=2",
        "train.micro_batch=2",
        "train.pool_size=4",
        "images.size=64",
        "model.tokenizer=fake",
    ]
    items += [f"{k}={v}" for k, v in over.items()]
    return load_config(None, items)


def _stream(cfg: Config, **kw: object) -> TrainStream:
    return TrainStream(
        cfg,
        rank=kw.pop("rank", 0),
        world_size=kw.pop("world", 1),
        tokenizer=synth.FakeTokenizer(),
        **kw,
    )  # type: ignore[arg-type]


def _take(it, n: int) -> list[dict]:
    return [next(it) for _ in range(n)]


def _sig(batch: dict) -> tuple:
    return (
        batch["meta"]["stream_id"],
        batch["meta"]["batch_index"],
        batch["input_ids"].numpy().tobytes(),
        batch["pixel_tiles"].numpy().tobytes(),
        batch["label_ids"].numpy().tobytes(),
        batch["label_pos"].numpy().tobytes(),
    )


def test_plan_partitions_shards_and_locates_cursors() -> None:
    shards = [ShardInfo(f"train/shard-{i:05d}.parquet", 3 + i % 4) for i in range(11)]
    plan = StreamPlan(shards, seed=1, n_streams=3, interleave=2, pool_size=4, micro_batch=2)
    for epoch in range(3):
        seen = [s.path for st in range(3) for g in plan.groups(st, epoch) for s in g]
        assert sorted(seen) == sorted(s.path for s in shards)
    assert plan.groups(0, 0) != plan.groups(0, 1)  # reshuffled per epoch
    g = plan.groups(1, 0)[0]
    rows = plan.row_order(g, 0)
    assert len(rows) == len(set(rows)) == sum(s.n_images for s in g)
    # locate() agrees with walking the structure batch by batch
    walk = []
    for epoch in range(2):
        for gi, grp in enumerate(plan.groups(2, epoch)):
            for pi, size in enumerate(plan.pool_sizes(sum(s.n_images for s in grp))):
                walk += [(epoch, gi, pi, b) for b in range(plan.n_batches(size))]
    for cursor, expected in enumerate(walk):
        p = plan.locate(2, cursor)
        assert (p.epoch, p.group, p.pool, p.batch) == expected


def test_batches_well_formed(hub: Path, tmp_path: Path) -> None:
    cfg = _cfg(hub, tmp_path)
    batches = _take(iter(_stream(cfg)), 30)
    streams = Counter(b["meta"]["stream_id"] for b in batches)
    assert set(streams) == {0, 1}
    for b in batches:
        B, L = b["input_ids"].shape
        assert B <= 2
        pos = b["label_pos"]
        assert (pos // L).max() < B
        # the label token is the token right after each label position
        flat = b["input_ids"].reshape(-1)
        assert torch.equal(flat[pos + 1], b["label_ids"])
        assert b["turn_weight"].numel() == int(b["label_turn"].max()) + 1
        n_img = int((b["image_grid_thw"][:, 1] * b["image_grid_thw"][:, 2]).sum())
        assert b["pixel_tiles"].shape == (n_img, 3, 16, 16)
        assert int(b["mm_token_type_ids"].sum()) == n_img // 4


@pytest.mark.parametrize("consumed", [37, 64])
def test_exact_resume(hub: Path, tmp_path: Path, consumed: int) -> None:
    cfg = _cfg(hub, tmp_path)
    reference = _take(iter(_stream(cfg)), consumed + 100)
    # a trainer consumed `consumed` batches, then crashed: cursors + rotation are all it kept
    cursors: dict[int, int] = {}
    for b in reference[:consumed]:
        cursors[b["meta"]["stream_id"]] = b["meta"]["batch_index"] + 1
    resumed = _stream(cfg, cursors=cursors, rotation=consumed % cfg.data.num_workers)
    after = _take(iter(resumed), 100)
    assert [_sig(b) for b in after] == [_sig(b) for b in reference[consumed:]]
    epochs = {b["meta"]["epoch"] for b in reference}
    assert len(epochs) >= 3  # the window crosses epoch boundaries


def test_dataloader_workers_match_in_process_order(hub: Path, tmp_path: Path) -> None:
    cfg = _cfg(hub, tmp_path)
    expected = _take(iter(_stream(cfg)), 16)
    loader = make_train_loader(
        _stream(
            load_config(
                None,
                [
                    f"data.hub_dataset=local:{hub}",
                    f"data.cache_dir={tmp_path / 'cache2'}",
                    "data.num_workers=2",
                    "train.micro_batch=2",
                    "train.pool_size=4",
                    "images.size=64",
                    "data.prefetch_factor=2",
                ],
            )
        ),
        pin_memory=False,
    )
    got = _take(iter(loader), 16)
    assert [_sig(b) for b in got] == [_sig(b) for b in expected]


def test_ranks_get_disjoint_streams(hub: Path, tmp_path: Path) -> None:
    cfg = _cfg(hub, tmp_path)
    r0 = _take(iter(_stream(cfg, rank=0, world=2)), 20)
    r1 = _take(iter(_stream(cfg, rank=1, world=2)), 20)
    assert {b["meta"]["stream_id"] for b in r0} == {0, 1}
    assert {b["meta"]["stream_id"] for b in r1} == {2, 3}
    first_epoch0 = {p for b in r0 if b["meta"]["epoch"] == 0 for p in b["meta"]["patch_ids"]}
    first_epoch1 = {p for b in r1 if b["meta"]["epoch"] == 0 for p in b["meta"]["patch_ids"]}
    assert not first_epoch0 & first_epoch1


def test_flaky_network_gives_identical_stream(hub: Path, tmp_path: Path) -> None:
    cfg = _cfg(hub, tmp_path)
    clean = _take(iter(_stream(cfg)), 40)
    flaky_cfg = _cfg(hub, tmp_path / "f")
    flaky = _stream(flaky_cfg, mirror_spec=f"flaky:0.4:local:{hub}")
    got = _take(iter(flaky), 40)
    assert [_sig(b) for b in got] == [_sig(b) for b in clean]


def test_shards_are_deleted_after_use(hub: Path, tmp_path: Path) -> None:
    cfg = _cfg(hub, tmp_path, **{"data.prefetch_shards": 1})
    _take(iter(_stream(cfg)), 60)
    cached = list((tmp_path / "cache").rglob("shard-*.parquet"))
    assert len(cached) <= 2 * (cfg.data.interleave + 1)


def test_training_stream_rejects_non_train_data(hub: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train split only"):
        _stream(_cfg(hub, tmp_path, **{"data.split": "val_small"}))
    bad = tmp_path / "badhub"
    shutil.copytree(hub, bad)
    victim = sorted((bad / "train").glob("*.parquet"))[0]
    shutil.copyfile(next((bad / "bench").glob("*.parquet")), victim)
    cfg = _cfg(bad, tmp_path / "b")
    with pytest.raises(AssertionError, match="non-train rows"):
        _take(iter(_stream(cfg)), 200)
