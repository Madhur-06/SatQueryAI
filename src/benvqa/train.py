"""Stage 3: LoRA training loop (``torchrun --nproc_per_node 2 -m benvqa.train --config ...``).

Per optimizer step:
1. Pull ``grad_accum`` micro-batches (time blocked on the loader -> ``data_wait_frac``).
2. One all-reduce of [turn-weight sum, conversations, QAs, tokens] gives the global
   normaliser ``W``. Each micro-batch loss is ``sum w_t * mean CE_t / (W / world)``
   (see ``model/loss.py``), and backward runs under ``no_sync`` except for the last.
3. Non-finite loss on any rank -> skip the step; ``nan_patience`` in a row -> reload the
   last checkpoint and halve the lr. Otherwise unscale, clip, WSD lr, step.
4. Rank 0 decides checkpoint / eval / stop (20-min timer, step interval, SIGTERM, session
   guard, planned images) and broadcasts the decision, so collectives stay aligned.

Resume (local ``LATEST`` first, else the mirror's ``ckpt/LATEST``) restores the adapter,
optimizer, GradScaler, schedule state, RNG, counters and the per-stream data cursors.
The data stream then continues exactly where it stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from benvqa.checkpoint import CheckpointManager
from benvqa.config import Config, load_config
from benvqa.data.collate import CATEGORIES, KINDS
from benvqa.data.render import Renderer
from benvqa.data.stream import (
    ImageMaker,
    TrainStream,
    default_cache_dir,
    load_quant_spec,
    make_train_loader,
)
from benvqa.evaluation import ScoreEvaluator, balanced_subset, load_split, summarize, vqa_questions
from benvqa.hub_sync import HubMirror, make_mirror
from benvqa.logging_utils import JsonlLogger, gpu_memory_gb, setup_logging
from benvqa.model.load import adapter_state, load_adapter_weights, load_model
from benvqa.model.loss import LabelPositionLM
from benvqa.schedule import ScheduleState, conversations_per_step, decay_now, lr_at, planned_steps
from benvqa.session_guard import SessionGuard

log = logging.getLogger("benvqa.train")


# ------------------------------------------------------------------------------ helpers
class Dist:
    def __init__(self) -> None:
        self.enabled = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            if self.local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"local rank {self.local_rank} but only "
                    f"{torch.cuda.device_count()} GPU(s) visible"
                )
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        if self.enabled:
            minutes = float(os.environ.get("BENVQA_DIST_TIMEOUT_MIN", "30"))
            dist.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo",
                timeout=datetime.timedelta(minutes=minutes),
            )

    @property
    def main(self) -> bool:
        return self.rank == 0

    def all_reduce(self, t: torch.Tensor, op: Any = None) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
        return t

    def broadcast_obj(self, obj: Any) -> Any:
        if not self.enabled:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def gather_obj(self, obj: Any) -> list[Any]:
        if not self.enabled:
            return [obj]
        out: list[Any] = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled:
            dist.destroy_process_group()


def get_tokenizer(cfg: Config) -> Any:
    if cfg.model.tokenizer == "fake":  # tests: tiny random models with a word-hash tokenizer
        from benvqa.testing import FakeTokenizer

        return FakeTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.model.tokenizer, local_files_only=cfg.model.local_files_only
    )


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 (global state of libraries)
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def set_rng_state(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])  # noqa: NPY002
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"])


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# -------------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: Config, args: argparse.Namespace, d: Dist) -> None:
        self.cfg, self.args, self.d = cfg, args, d
        self.guard = SessionGuard(cfg.session.max_session_hours)
        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.log_path = self.ckpt_dir.parent / "train_log.jsonl"
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg.model.dtype]
        self.amp = d.device.type == "cuda" and self.dtype != torch.float32
        mirror = None
        if cfg.checkpoint.hub_repo:
            kw = (
                {}
                if cfg.checkpoint.hub_repo.startswith("local:")
                else {"repo_type": "model", "create": d.main, "private": True}
            )
            mirror = make_mirror(cfg.checkpoint.hub_repo, **kw)
        self.ckpt = CheckpointManager(
            self.ckpt_dir,
            mirror if d.main else None,
            cfg.checkpoint.keep_local,
            cfg.checkpoint.keep_hub,
            extra_files={"train_log.jsonl": self.log_path},
        )

        # ---- resume state
        state = self._find_resume()
        self.state: dict[str, Any] = state["state"] if state else {}
        if state:
            # Restore the dataset revision first: the saved data_hash includes the revision
            # the first run resolved, while a resumed config usually has none. An explicit,
            # different revision is a data change and fails the check below.
            if cfg.data.revision is None:
                cfg.data.revision = self.state.get("data_revision")
            self._check_compatible(self.state)
        elif cfg.data.hub_dataset.startswith("hub:") and cfg.data.revision is None and d.main:
            cfg.data.revision = HubMirror(cfg.data.hub_dataset[4:]).head_revision()
        cfg.data.revision = d.broadcast_obj(cfg.data.revision)

        # ---- model / optimizer
        torch.manual_seed(cfg.seed + d.rank)
        self.loaded = load_model(cfg, d.device)
        self.lm = LabelPositionLM(self.loaded.model, self.loaded.base)
        # from_pretrained returns eval mode, and HF applies gradient checkpointing only when
        # training: without this, activations of every layer stay alive (M5a: 7.8 vs 3.3 GB).
        self.lm.train()
        if cfg.model.gradient_checkpointing and cfg.model.backend == "hf":
            assert self.loaded.base.is_gradient_checkpointing, "gradient checkpointing is off"
        self.params = [p for p in self.lm.parameters() if p.requires_grad]
        if state:
            load_adapter_weights(
                self.loaded.model, {k: v.to(d.device) for k, v in state["adapter"].items()}
            )
        self.ddp: Any = self.lm
        if d.enabled:
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                self.lm,
                device_ids=[d.local_rank] if d.device.type == "cuda" else None,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=cfg.train.ddp_static_graph,
            )
        self.opt = torch.optim.AdamW(
            self.params,
            lr=cfg.train.lr,
            betas=cfg.train.adam_betas,
            weight_decay=cfg.train.weight_decay,
            fused=d.device.type == "cuda",
        )
        self.scaler = torch.amp.GradScaler(
            d.device.type, enabled=self.amp and self.dtype == torch.float16
        )
        self.sched = ScheduleState(**self.state.get("schedule", {}))
        self.step = int(self.state.get("step", 0))
        self.counters: dict[str, float] = dict(self.state.get("counters", {}))
        self.cursors: dict[int, int] = {int(k): v for k, v in self.state.get("cursors", {}).items()}
        consumed = self.state.get("consumed_by_rank", [0] * d.world)
        self.consumed_rank = int(consumed[d.rank])
        self.best = float(self.state.get("best_mean_oa", -1.0))
        if state:
            self.opt.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            set_rng_state(state["rng"][d.rank])
            log.info("resumed from step %d (%s)", self.step, self.state.get("checkpoint"))
        if args.decay_now:
            sec = float(self.state.get("metrics", {}).get("sec_per_step", 30.0))
            fit = int(max(1, (self.guard.remaining_s() - 1800) / sec))
            self.sched = decay_now(self.step, cfg, d.world, fit)
            log.info("decay now: %s", self.sched)

        # ---- data
        nw = max(1, cfg.data.num_workers)
        mine = {s: c for s, c in self.cursors.items() if s // nw == d.rank}
        self.tokenizer = get_tokenizer(cfg)
        self.stream = TrainStream(
            cfg,
            d.rank,
            d.world,
            cursors=mine,
            rotation=self.consumed_rank % nw,
            tokenizer=self.tokenizer,
        )
        self.loader = iter(make_train_loader(self.stream, pin_memory=d.device.type == "cuda"))
        self.evaluator: ScoreEvaluator | None = None
        self.val_questions: list = []
        self.logger = JsonlLogger(self.log_path, cfg.model_dump(), run_name="") if d.main else None
        self.window = self._new_window()
        self.last_ckpt_time = time.time()
        self.nan_streak = 0

    # ---------------------------------------------------------------- resume helpers
    def _find_resume(self) -> dict[str, Any] | None:
        if self.args.resume == "none":
            return None
        path = None
        if self.d.main:
            path = self.ckpt.resume_path()
        path = self.d.broadcast_obj(str(path) if path else None)
        if path is None:
            return None
        self.d.barrier()
        loaded = CheckpointManager.load(Path(path), "cpu")
        loaded["state"]["checkpoint"] = path
        return loaded

    def _check_compatible(self, state: dict[str, Any]) -> None:
        if state.get("world_size") != self.d.world:
            raise RuntimeError(
                f"checkpoint was trained with world_size={state.get('world_size')}, now "
                f"{self.d.world}: data cursors would not map. Use the same number of GPUs."
            )
        if state.get("data_hash") != self.cfg.data_hash() and not self.args.allow_data_change:
            raise RuntimeError(
                "data/conversation config changed since the checkpoint "
                "(data_hash mismatch); pass --allow-data-change to continue anyway"
            )

    # ---------------------------------------------------------------- metrics window
    def _new_window(self) -> dict[str, Any]:
        n = len(CATEGORIES) * len(KINDS)
        return {
            "t0": time.time(),
            "wait": 0.0,
            "steps": 0,
            "conv": 0.0,
            "qa": 0.0,
            "tok": 0.0,
            "loss": 0.0,
            "gnorm": 0.0,
            "skipped": 0,
            "cat_loss": torch.zeros(n, device=self.d.device),
            "cat_n": torch.zeros(n, device=self.d.device),
            "cat_correct": torch.zeros(n, device=self.d.device),
        }

    # ---------------------------------------------------------------- main loop
    def train_step(self) -> dict[str, Any]:
        cfg, d = self.cfg, self.d
        t_start = time.perf_counter()
        mbs, wait = [], 0.0
        for _ in range(cfg.train.grad_accum):
            t = time.perf_counter()
            mbs.append(next(self.loader))
            wait += time.perf_counter() - t
        tot = torch.tensor(
            [
                sum(float(b["turn_weight"].sum()) for b in mbs),
                sum(b["meta"]["n_conversations"] for b in mbs),
                sum(b["meta"]["n_qa"] for b in mbs),
                sum(b["meta"]["n_tokens"] for b in mbs),
            ],
            device=d.device,
            dtype=torch.float64,
        )
        d.all_reduce(tot)
        weight_norm = float(tot[0]) / d.world
        loss_sum = torch.zeros((), device=d.device)
        nk = len(KINDS)
        for k, b in enumerate(mbs):
            b = to_device(b, d.device)
            last = k == len(mbs) - 1
            sync = contextlib.nullcontext() if last or not d.enabled else self.ddp.no_sync()
            with sync, torch.autocast(d.device.type, dtype=self.dtype, enabled=self.amp):
                loss, aux = self.ddp(b, weight_norm)
            self.scaler.scale(loss).backward()
            loss_sum += loss.detach().float()
            cell = b["turn_category"] * nk + b["turn_kind"]
            w = self.window
            w["cat_loss"].index_add_(0, cell, aux["turn_loss"].float())
            w["cat_n"].index_add_(0, cell, torch.ones_like(aux["turn_loss"], dtype=torch.float))
            w["cat_correct"].index_add_(0, cell, aux["turn_correct"].float())
            self.cursors[b["meta"]["stream_id"]] = b["meta"]["batch_index"] + 1
            self.consumed_rank += 1
        step_loss = d.all_reduce(loss_sum.clone()) / d.world
        finite = bool(torch.isfinite(step_loss))
        if not finite:
            self.opt.zero_grad(set_to_none=True)
            self.nan_streak += 1
            self.window["skipped"] += 1
            log.warning(
                "non-finite loss at step %d (%d in a row): step skipped", self.step, self.nan_streak
            )
            if self.nan_streak >= cfg.train.nan_patience:
                self._recover_from_nan()
            gnorm = float("nan")
        else:
            self.nan_streak = 0
            self.scaler.unscale_(self.opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.train.max_grad_norm))
            lr = lr_at(self.step, cfg, d.world, self.sched)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
        c = self.counters
        c["conversations"] = c.get("conversations", 0) + float(tot[1])
        c["qa"] = c.get("qa", 0) + float(tot[2])
        c["tokens"] = c.get("tokens", 0) + float(tot[3])
        w = self.window
        w["wait"] += wait
        w["steps"] += 1
        w["conv"] += float(tot[1])
        w["qa"] += float(tot[2])
        w["tok"] += float(tot[3])
        if finite:
            w["loss"] += float(step_loss)
            w["gnorm"] += gnorm
        return {"time": time.perf_counter() - t_start}

    def _recover_from_nan(self) -> None:
        path = self.ckpt.latest_local()
        path = self.d.broadcast_obj(str(path) if path else None)
        self.sched.lr_scale *= 0.5
        if path:
            st = CheckpointManager.load(Path(path), "cpu")
            load_adapter_weights(
                self.loaded.model, {k: v.to(self.d.device) for k, v in st["adapter"].items()}
            )
            self.opt.load_state_dict(st["optimizer"])
            self.scaler.load_state_dict(st["scaler"])
        log.error(
            "NaN recovery: reloaded %s, lr scale now %.3g (data continues)",
            path,
            self.sched.lr_scale,
        )
        self.nan_streak = 0

    def log_window(self) -> None:
        d, w, cfg = self.d, self.window, self.cfg
        for key in ("cat_loss", "cat_n", "cat_correct"):
            d.all_reduce(w[key])
        dt = time.time() - w["t0"]
        if d.main and self.logger is not None:
            per_cat: dict[str, float] = {}
            acc: dict[str, float] = {}
            nk = len(KINDS)
            for i in range(len(w["cat_n"])):
                n = float(w["cat_n"][i])
                if n:
                    name = f"{KINDS[i % nk]}:{CATEGORIES[i // nk]}"
                    per_cat[name] = round(float(w["cat_loss"][i]) / n, 4)
                    if KINDS[i % nk] != "caption":
                        acc[name] = round(float(w["cat_correct"][i]) / n, 4)
            good = max(1, w["steps"] - w["skipped"])
            conv_s = w["conv"] / dt
            remaining = max(0.0, cfg.train.planned_images - self.counters.get("conversations", 0))
            self.logger.write(
                {
                    "step": self.step,
                    "loss": round(w["loss"] / good, 5),
                    "lr": lr_at(max(0, self.step - 1), cfg, d.world, self.sched),
                    "grad_norm": round(w["gnorm"] / good, 4),
                    "scaler_scale": self.scaler.get_scale(),
                    "skipped_steps": w["skipped"],
                    "loss_by_category": per_cat,
                    "train_acc": acc,
                    "conversations_per_s": round(conv_s, 3),
                    "qa_per_s": round(w["qa"] / dt, 2),
                    "tokens_per_s": round(w["tok"] / dt, 1),
                    "sec_per_step": round(dt / w["steps"], 3),
                    "data_wait_frac": round(w["wait"] / dt, 4),
                    "gpu_mem_gb": gpu_memory_gb(),
                    "conversations_seen": self.counters.get("conversations", 0),
                    "qa_seen": self.counters.get("qa", 0),
                    "tokens_seen": self.counters.get("tokens", 0),
                    "eta_hours": round(remaining / max(conv_s, 1e-9) / 3600, 2),
                    "session_hours": round(self.guard.elapsed_s() / 3600, 3),
                }
            )
            if w["wait"] / dt > 0.05 and getattr(self, "_logged_once", False):  # 1st: worker start
                log.warning(
                    "data_wait_frac %.1f%% > 5%%: the loader is a bottleneck", 100 * w["wait"] / dt
                )
            log.info(
                "step %d loss %.4f %.2f conv/s wait %.1f%%",
                self.step,
                w["loss"] / good,
                conv_s,
                100 * w["wait"] / dt,
            )
        self._last_sec_per_step = dt / max(1, w["steps"])
        self._logged_once = True
        self.window = self._new_window()

    # ---------------------------------------------------------------- checkpoint / eval
    def save(self, reason: str) -> Path | None:
        d = self.d
        cursors_all = d.gather_obj(self.cursors)
        consumed_all = d.gather_obj(self.consumed_rank)
        rng_all = d.gather_obj(rng_state())
        path = None
        if d.main:
            merged: dict[int, int] = {}
            for cur in cursors_all:
                merged.update(cur)
            state = {
                "step": self.step,
                "counters": self.counters,
                "cursors": merged,
                "consumed_by_rank": consumed_all,
                "world_size": d.world,
                "data_hash": self.cfg.data_hash(),
                "data_revision": self.cfg.data.revision,
                "schedule": self.sched.to_dict(),
                "best_mean_oa": self.best,
                "reason": reason,
                "config": self.cfg.model_dump(),
                "planned_steps": planned_steps(self.cfg, d.world),
                "metrics": {"sec_per_step": getattr(self, "_last_sec_per_step", None)},
                "backend": self.loaded.backend,
            }
            peft_cfg = self.loaded.model.peft_config["default"].to_dict()
            path = self.ckpt.save(
                self.step,
                {
                    "adapter": adapter_state(self.loaded.model),
                    "adapter_config": {
                        k: (sorted(v) if isinstance(v, set) else v) for k, v in peft_cfg.items()
                    },
                    "optimizer": self.opt.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "rng": rng_all,
                    "state": state,
                },
            )
        self.last_ckpt_time = time.time()
        d.barrier()
        return path

    def setup_eval(self) -> None:
        cfg, d = self.cfg, self.d
        mirror_spec = cfg.data.hub_dataset
        kw = {} if mirror_spec.startswith("local:") else {"revision": cfg.data.revision}
        mirror = make_mirror(mirror_spec, **kw)
        cache = Path(cfg.data.cache_dir) / f"eval-r{d.rank}"
        rows = load_split(mirror, cache, cfg.eval.split)
        qs = balanced_subset(vqa_questions(rows), cfg.eval.val_questions, seed=cfg.seed)
        for q in qs:
            q.cube = q.cube.copy()  # drop references to the full shard tables
        self.val_questions = qs[d.rank :: d.world]
        renderer = Renderer(self.tokenizer, cfg.images.size, cfg.train.max_len)
        comps = tuple(cfg.eval.composites or cfg.images.composites)
        self.evaluator = ScoreEvaluator(
            renderer,
            ImageMaker(cfg, load_quant_spec(mirror, cache)),
            comps,
            cfg.eval.batch_size,
            cfg.train.pad_to_multiple,
        )

    def evaluate(self) -> dict[str, Any] | None:
        if self.evaluator is None:
            self.setup_eval()
        assert self.evaluator is not None
        self.lm.eval()
        t0 = time.time()
        recs = self.evaluator.run(self.lm, self.val_questions, self.d.device)
        self.lm.train()
        allrecs = [r for part in self.d.gather_obj(recs) for r in part]
        if not self.d.main:
            return None
        summary = summarize(allrecs)
        summary["eval_seconds"] = round(time.time() - t0, 1)
        if self.logger is not None:
            self.logger.write({"step": self.step, "val": summary})
        log.info(
            "val step %d: binary %.2f mcq %.2f mean %.2f",
            self.step,
            summary.get("binary", {}).get("OA", 0),
            summary.get("mcq", {}).get("OA", 0),
            summary["mean_OA"],
        )
        return summary

    def eval_and_checkpoint(self, reason: str) -> None:
        summary = self.evaluate()
        improved = False
        if self.d.main and summary is not None and summary["mean_OA"] > self.best:
            self.best, improved = summary["mean_OA"], True
        path = self.save(reason)
        if self.d.main and improved and path is not None:
            self.ckpt.save_best(path, {"step": self.step, **summary})

    # ---------------------------------------------------------------- control
    def decide(self, step_time: float) -> tuple[bool, bool, str | None]:
        cfg = self.cfg
        do_eval = do_ckpt = False
        stop: str | None = None
        if self.d.main:
            if cfg.train.max_steps is not None and self.step >= cfg.train.max_steps:
                stop = "max_steps"
            elif self.counters.get("conversations", 0) >= cfg.train.planned_images:
                stop = "planned images reached"
            else:
                stop = self.guard.stop_reason()
                # stop early enough to eval + save + upload within the session
                if stop is None and self.guard.remaining_s() < max(600, 3 * step_time):
                    stop = "session time guard"
            do_eval = (
                cfg.eval.every_steps > 0 and self.step > 0 and self.step % cfg.eval.every_steps == 0
            )
            do_ckpt = (self.step % cfg.checkpoint.every_steps == 0 and self.step > 0) or (
                time.time() - self.last_ckpt_time >= 60 * cfg.checkpoint.every_minutes
            )
        return self.d.broadcast_obj((do_eval, do_ckpt, stop))

    def run(self) -> None:
        cfg = self.cfg
        log.info(
            "training: %d planned steps (%d conversations/step), starting at step %d",
            planned_steps(cfg, self.d.world),
            conversations_per_step(cfg, self.d.world),
            self.step,
        )
        stop = None
        try:
            while stop is None:
                info = self.train_step()
                if self.step % cfg.train.log_every == 0 or self.window["steps"] >= 50:
                    self.log_window()
                if os.environ.get("BENVQA_FAULT_EXIT_AT_STEP") == str(self.step):
                    os._exit(86)  # fault injection (tests): die like a killed session, no cleanup
                do_eval, do_ckpt, stop = self.decide(info["time"])
                if stop is not None:
                    break
                if do_eval:
                    self.eval_and_checkpoint("eval")
                elif do_ckpt:
                    self.save("interval")
            log.info("stopping: %s", stop)
            if self.window["steps"]:
                self.log_window()
            self.eval_and_checkpoint(f"stop: {stop}")
        finally:
            if self.d.main:
                self.ckpt.finalize()
            self.d.barrier()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=None, help="YAML config (else defaults + --set)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.max_steps=20")
    ap.add_argument("--resume", choices=["auto", "none"], default="auto")
    ap.add_argument("--decay-now", action="store_true", help="start the WSD decay at this session")
    ap.add_argument("--allow-data-change", action="store_true")
    args = ap.parse_args(argv)
    d = Dist()
    setup_logging(d.rank)
    cfg = load_config(args.config, args.set)
    if cfg.model.backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers is used)
    cfg.data.cache_dir = default_cache_dir(cfg.data.cache_dir)
    random.seed(cfg.seed + d.rank)
    np.random.seed(cfg.seed + d.rank)  # noqa: NPY002 (seeds third-party code)
    try:
        Trainer(cfg, args, d).run()
    finally:
        d.close()


if __name__ == "__main__":
    main()
