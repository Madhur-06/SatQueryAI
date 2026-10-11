"""Stage 3 on CPU with a tiny random Qwen3.5: loss equivalence, right-padding invariance,
schedule, checkpoints, hard kill -> exact resume, and 2-process DDP (gloo)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from benvqa import testing as synth  # noqa: E402
from benvqa.checkpoint import CheckpointManager  # noqa: E402
from benvqa.config import load_config  # noqa: E402
from benvqa.data import build_shards as bs  # noqa: E402
from benvqa.data.stream import TrainStream  # noqa: E402
from benvqa.hub_sync import LocalMirror  # noqa: E402
from benvqa.model.load import load_model  # noqa: E402
from benvqa.model.loss import LabelPositionLM, reference_loss  # noqa: E402
from benvqa.schedule import ScheduleState, decay_now, lr_at, planned_steps  # noqa: E402
from benvqa.session_guard import SessionGuard  # noqa: E402

SRC = str(Path(__file__).resolve().parents[2] / "src")


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("train")
    images = synth.make_images({"train": 24, "validation": 6, "bench": 1}, seed=5)
    parquet = synth.write_parquet(root / "ann.parquet", images, seed=5)
    s2, s1 = synth.write_tars(root / "tars", images, extra=0)
    bs.build(
        Namespace(
            parquet=parquet,
            mirror=f"local:{root / 'hub'}",
            private=False,
            work_dir=root / "work",
            s1_url=str(s1),
            s2_url=str(s2),
            val_small="all",
            include_full_val=False,
            seed=0,
            shard_size=3,
            row_group=3,
            buckets=2,
            zstd_level=1,
            max_members=None,
            skip_s1=False,
            connections=1,
            chunk_mb=1,
        )
    )
    return {"root": root, "hub": root / "hub", "model": synth.write_tiny_model(root / "model")}


def _overrides(env: dict[str, Path], run: Path, **extra: object) -> list[str]:
    items = {
        "model.name": env["model"],
        "model.tokenizer": "fake",
        "model.backend": "hf",
        "model.dtype": "float32",
        "model.gradient_checkpointing": False,
        "data.hub_dataset": f"local:{env['hub']}",
        "data.cache_dir": run / "cache",
        "data.num_workers": 2,
        "data.prefetch_factor": 2,
        "images.size": 64,
        "train.micro_batch": 2,
        "train.grad_accum": 2,
        "train.pool_size": 4,
        "train.log_every": 1,
        "train.planned_images": 400,
        "train.schedule": "{type: wsd, warmup_steps: 2, decay_fraction: 0.5}",
        "checkpoint.dir": run / "ckpt",
        "checkpoint.hub_repo": f"local:{run / 'mirror'}",
        "checkpoint.every_steps": 4,
        "eval.every_steps": 0,
        "eval.val_questions": 8,
        "eval.batch_size": 4,
    }
    items.update(extra)
    return [f"{k}={v}" for k, v in items.items()]


def _batches(env: dict[str, Path], tmp: Path, n: int) -> list[dict]:
    cfg = load_config(None, _overrides(env, tmp))
    it = iter(TrainStream(cfg, 0, 1, tokenizer=synth.FakeTokenizer()))
    return [next(it) for _ in range(n)]


def _tiny_lm(env: dict[str, Path], tmp: Path) -> LabelPositionLM:
    cfg = load_config(None, _overrides(env, tmp))
    loaded = load_model(cfg, torch.device("cpu"))
    with torch.no_grad():  # LoRA B starts at zero; perturb so adapters actually matter
        for p in loaded.model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))
    return LabelPositionLM(loaded.model, loaded.base).eval()


# ------------------------------------------------------------------------------ loss
def test_label_position_loss_matches_full_logits(env: dict[str, Path], tmp_path: Path) -> None:
    lm = _tiny_lm(env, tmp_path)
    for b in _batches(env, tmp_path, 4):
        with torch.no_grad():
            ours, _ = lm(b, weight_norm=3.0)
            ref = reference_loss(lm.base, b, weight_norm=3.0)
        assert torch.allclose(ours, ref, rtol=1e-5, atol=1e-6), (ours, ref)


def test_right_padding_does_not_change_samples(env: dict[str, Path], tmp_path: Path) -> None:
    """GatedDeltaNet is recurrent: right padding must not leak into real positions."""
    lm = _tiny_lm(env, tmp_path)
    b = next(
        x
        for x in _batches(env, tmp_path, 10)
        if x["input_ids"].shape[0] == 2
        and (x["attention_mask"].sum(1)[0] != x["attention_mask"].sum(1)[1])
    )
    with torch.no_grad():
        h_batch = lm.hidden_states(b, torch.float32)
    lens = b["attention_mask"].sum(1)
    n_img = [int(x) for x in (b["input_ids"] == 5).sum(1) // 4]  # images per sample (64x64)
    tile_split = torch.split(b["pixel_tiles"], [n * 16 for n in n_img])
    grid_split = torch.split(b["image_grid_thw"], n_img)
    for i in range(2):
        n = int(lens[i])
        single = {
            "input_ids": b["input_ids"][i : i + 1, :n],
            "attention_mask": b["attention_mask"][i : i + 1, :n],
            "mm_token_type_ids": b["mm_token_type_ids"][i : i + 1, :n],
            "pixel_tiles": tile_split[i],
            "image_grid_thw": grid_split[i],
        }
        with torch.no_grad():
            h_one = lm.hidden_states(single, torch.float32)
        torch.testing.assert_close(h_batch[i, :n], h_one[0], atol=2e-5, rtol=1e-4)


# -------------------------------------------------------------------------- schedule
def test_wsd_schedule_shape() -> None:
    cfg = load_config(
        None,
        [
            "train.planned_images=3200",
            "train.micro_batch=2",
            "train.grad_accum=8",
            "train.schedule={warmup_steps: 10, decay_fraction: 0.15}",
        ],
    )
    total = planned_steps(cfg, 2)
    assert total == 100
    lrs = [lr_at(s, cfg, 2, ScheduleState()) for s in range(total + 5)]
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[9] == pytest.approx(1e-4)
    assert lrs[50] == lrs[84] == pytest.approx(1e-4)  # stable until 85% of planned steps
    assert lrs[85] == pytest.approx(1e-4)
    assert lrs[86] < 1e-4
    assert lrs[99] < 1e-5
    assert lrs[104] == 0.0
    st = decay_now(40, cfg, 2, steps_fit_in_session=10)
    assert lr_at(39, cfg, 2, st) == pytest.approx(1e-4)
    assert lr_at(45, cfg, 2, st) == pytest.approx(5e-5)
    assert lr_at(50, cfg, 2, st) == 0.0
    half = ScheduleState(lr_scale=0.5)
    assert lr_at(50, cfg, 2, half) == pytest.approx(5e-5)


# ------------------------------------------------------------------------ checkpoints
def _payload(step: int) -> dict:
    return {
        "adapter": {"w": torch.full((3,), float(step))},
        "adapter_config": {"r": 4},
        "optimizer": {"s": step},
        "scaler": {},
        "rng": [],
        "state": {"step": step},
    }


def test_checkpoint_rotation_and_remote_resume(tmp_path: Path) -> None:
    mirror = LocalMirror(tmp_path / "mirror")
    mgr = CheckpointManager(tmp_path / "ckpt", mirror, keep_local=2, keep_remote=3)
    for step in (1, 2, 3, 4, 5):
        mgr.save(step, _payload(step))
        mgr.uploader.wait()  # type: ignore[union-attr]
    assert sorted(p.name for p in (tmp_path / "ckpt").glob("step-*")) == [
        "step-000004",
        "step-000005",
    ]
    remote = sorted({p.split("/")[1] for p in mirror.list("ckpt/step-")})
    assert remote == ["step-000003", "step-000004", "step-000005"]
    assert (tmp_path / "mirror/ckpt/LATEST").read_text() == "step-000005"
    assert not list((tmp_path / "ckpt").glob("*.tmp"))
    # a fresh machine: nothing local, resume from the mirror
    fresh = CheckpointManager(tmp_path / "fresh", mirror)
    path = fresh.resume_path()
    assert path is not None
    assert path.name == "step-000005"
    assert CheckpointManager.load(path)["state"]["step"] == 5


def test_busy_uploader_keeps_checkpoint_local_then_finalize(tmp_path: Path) -> None:
    import threading

    gate = threading.Event()

    class Slow(LocalMirror):
        def upload(self, files, message):  # type: ignore[no-untyped-def]
            gate.wait(10)
            super().upload(files, message)

    mirror = Slow(tmp_path / "mirror")
    mgr = CheckpointManager(tmp_path / "ckpt", mirror, keep_local=1)
    mgr.save(1, _payload(1))  # upload starts and blocks
    mgr.save(2, _payload(2))  # uploader busy: local only
    mgr.save(3, _payload(3))
    assert (tmp_path / "ckpt/step-000001").exists()  # in flight: not pruned
    gate.set()
    mgr.finalize()
    assert (tmp_path / "mirror/ckpt/LATEST").read_text() == "step-000003"


def test_session_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal
    import time

    monkeypatch.setenv("SESSION_START_EPOCH", str(time.time() - 3600 * 11.2))
    g = SessionGuard(max_hours=11.3)
    assert 0 < g.remaining_s() < 400
    assert g.stop_reason() is None
    g._handle(signal.SIGTERM, None)
    assert g.stop_reason() == "signal SIGTERM"
    assert SessionGuard(max_hours=11.3, start_epoch=time.time() - 3600 * 12).stop_reason()


# ------------------------------------------------------------------ end-to-end runs
def _run(
    env: dict[str, Path],
    run: Path,
    *extra: str,
    fault: int | None = None,
    nproc: int = 1,
    steps: int = 8,
) -> subprocess.CompletedProcess:
    over = _overrides(env, run, **{"train.max_steps": steps})
    run.mkdir(parents=True, exist_ok=True)
    e = {
        **os.environ,
        "PYTHONPATH": SRC,
        "CUDA_VISIBLE_DEVICES": "-1",  # "" deletes the variable on Windows
        "BENVQA_DIST_TIMEOUT_MIN": "3",
        "HF_HUB_OFFLINE": "1",
        "SESSION_START_EPOCH": "",
        "USE_LIBUV": "0",
    }
    if fault is not None:
        e["BENVQA_FAULT_EXIT_AT_STEP"] = str(fault)
    cmd = [sys.executable, "-m", "benvqa.train", "--set", *over, *extra]
    if nproc == 1:
        return subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=600)
    # Launch the ranks directly (env:// rendezvous). torchrun's own rendezvous needs libuv,
    # which the Windows PyTorch build lacks. On Kaggle/Linux the notebook uses torchrun.
    procs = []
    for rank in range(nproc):
        env_r = {
            **e,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(nproc),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29533",
        }
        log = (run / f"rank{rank}.log").open("w")  # files, not pipes: a full pipe deadlocks DDP
        procs.append((subprocess.Popen(cmd, env=env_r, stdout=log, stderr=subprocess.STDOUT), log))
    import time

    deadline = time.time() + 600
    try:
        while any(p.poll() is None for p, _ in procs) and time.time() < deadline:
            if any(p.poll() not in (None, 0) for p, _ in procs):  # one rank died: stop the rest
                break
            time.sleep(0.5)
        codes = [p.poll() if p.poll() is not None else -9 for p, _ in procs]
    finally:
        for p, log in procs:
            if p.poll() is None:
                p.kill()
            log.close()
    text = "".join((run / f"rank{r}.log").read_text() for r in range(nproc))
    return subprocess.CompletedProcess(cmd, max(codes), "", text)


def _log(run: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in (run / "train_log.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if "loss" in rec:
            out[rec["step"]] = rec  # the last occurrence of a step wins (re-done after resume)
    return out


def test_hard_kill_then_exact_resume(env: dict[str, Path], tmp_path: Path) -> None:
    a = _run(env, tmp_path / "a")
    assert a.returncode == 0, a.stderr[-3000:]
    b1 = _run(env, tmp_path / "b", fault=6)  # dies after step 6; last checkpoint is step 4
    assert b1.returncode == 86, b1.stderr[-3000:]
    assert (tmp_path / "b/ckpt/LATEST").read_text() == "step-000004"
    b2 = _run(env, tmp_path / "b")
    assert b2.returncode == 0, b2.stderr[-3000:]
    la, lb = _log(tmp_path / "a"), _log(tmp_path / "b")
    assert sorted(la) == sorted(lb) == list(range(1, 9))
    for step in range(1, 9):
        assert lb[step]["loss"] == pytest.approx(la[step]["loss"], rel=1e-5), step
        assert lb[step]["lr"] == pytest.approx(la[step]["lr"]), step
    sa = CheckpointManager.load(tmp_path / "a/ckpt/step-000008")
    sb = CheckpointManager.load(tmp_path / "b/ckpt/step-000008")
    assert sa["state"]["cursors"] == sb["state"]["cursors"]
    assert sa["state"]["counters"] == sb["state"]["counters"]
    for k, v in sa["adapter"].items():
        torch.testing.assert_close(sb["adapter"][k], v, rtol=1e-5, atol=1e-7)


def test_resume_from_mirror_on_fresh_machine(env: dict[str, Path], tmp_path: Path) -> None:
    run = tmp_path / "c"
    assert _run(env, run, steps=4).returncode == 0
    import shutil

    shutil.rmtree(run / "ckpt")  # the Kaggle session's /kaggle/working is gone
    (run / "train_log.jsonl").unlink()
    r = _run(env, run, steps=6)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "resumed from step 4" in r.stderr
    assert sorted(_log(run)) == [1, 2, 3, 4, 5, 6]  # the log came back from the mirror too


def test_ddp_two_processes(env: dict[str, Path], tmp_path: Path) -> None:
    run = tmp_path / "ddp"
    r = _run(env, run, nproc=2, steps=4)
    assert r.returncode == 0, r.stderr[-4000:]
    state = CheckpointManager.load(run / "ckpt/step-000004")["state"]
    assert state["world_size"] == 2
    assert sorted(map(int, state["cursors"])) == [0, 1, 2, 3]
    # 4 steps x 2 ranks x 2 micro-batches x <=2 conversations (odd pools give a batch of 1)
    assert 24 <= state["counters"]["conversations"] <= 32


def test_eval_bench_zero_shot_and_finetuned(env: dict[str, Path], tmp_path: Path) -> None:
    run = tmp_path / "e"
    assert _run(env, run, steps=4).returncode == 0
    over = _overrides(env, run)
    e = {**os.environ, "PYTHONPATH": SRC, "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1"}
    out = tmp_path / "reports"
    cmd = [sys.executable, "-m", "benvqa.eval_bench", "--set", *over, "--runs", "a,c",
           "--split", "bench", "--captions", "1", "--max-new-tokens", "5", "--out", str(out),
           "--adapter", f"local:{run / 'mirror'}"]  # fmt: skip
    r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    a, c = (json.loads((out / f"bench_{k}.json").read_text()) for k in "ac")
    assert a["adapter"] is None
    assert c["adapter"]["source"].startswith("local:")
    assert a["summary"]["n"] == c["summary"]["n"] > 0
    assert a["composites"] == ["rgb"]
    assert len(c["composites"]) == 3
    md = (out / "bench_results.md").read_text()
    assert "RS-InternVL" in md
    assert "(c)" in md
    assert (out / "reliability_c.svg").read_text().startswith("<svg")
