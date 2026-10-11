"""Crash-proof checkpoints: atomic local writes + asynchronous mirror to the Hub.

Layout (local ``checkpoint.dir`` and remote ``ckpt/``)::

    step-000250/  adapter_model.safetensors  adapter_config.json  optimizer.pt  scaler.pt
                  trainer_state.json  rng.pt
    LATEST        "step-000250"
    best/         adapter_model.safetensors  adapter_config.json  trainer_state.json

A checkpoint is written to ``step-XXXXXX.tmp/``, every file is fsynced, then the folder is
renamed atomically and ``LATEST`` is replaced atomically. The mirror upload (folder +
LATEST + train log, one commit) runs on a background thread. If the uploader is still busy,
the new checkpoint stays local-only. ``finalize()`` at session end waits and uploads the
newest checkpoint if it was skipped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch

from benvqa.hub_sync import AsyncUploader, Mirror

log = logging.getLogger(__name__)
STEP_RE = re.compile(r"^step-(\d{6})$")
ADAPTER = "adapter_model.safetensors"


def _fsync(path: Path) -> None:
    with path.open("r+b") as fh:  # Windows needs a writable handle for fsync
        os.fsync(fh.fileno())


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync(tmp)
    os.replace(tmp, path)


def step_name(step: int) -> str:
    return f"step-{step:06d}"


class CheckpointManager:
    def __init__(
        self,
        root: Path,
        mirror: Mirror | None,
        keep_local: int = 2,
        keep_remote: int = 3,
        remote_prefix: str = "ckpt",
        extra_files: dict[str, Path] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mirror = mirror
        self.keep_local, self.keep_remote = keep_local, keep_remote
        self.prefix = remote_prefix
        self.extra_files = extra_files or {}
        self.uploader = AsyncUploader(mirror, "ckpt-uploader") if mirror is not None else None
        self.last_saved: str | None = None
        self.last_uploaded: str | None = None
        self._uploading: str | None = None
        """Folder currently being uploaded (never pruned locally while in flight)."""

    # ------------------------------------------------------------------------ writing
    def save(self, step: int, payload: dict[str, Any]) -> Path:
        """``payload``: adapter (state dict), adapter_config (dict), optimizer, scaler, rng
        (picklable objects) and state (JSON-able dict)."""
        from safetensors.torch import save_file

        name = step_name(step)
        final = self.root / name
        tmp = self.root / f"{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        save_file(payload["adapter"], str(tmp / ADAPTER))
        (tmp / "adapter_config.json").write_text(
            json.dumps(payload["adapter_config"], indent=1, default=str)
        )
        torch.save(payload["optimizer"], tmp / "optimizer.pt")
        torch.save(payload["scaler"], tmp / "scaler.pt")
        torch.save(payload["rng"], tmp / "rng.pt")
        (tmp / "trainer_state.json").write_text(json.dumps(payload["state"], indent=1, default=str))
        for f in tmp.iterdir():
            _fsync(f)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        _atomic_text(self.root / "LATEST", name)
        self.last_saved = name
        self._prune_local()
        self._upload(name)
        log.info("checkpoint %s saved", name)
        return final

    def save_best(self, src: Path, metrics: dict[str, Any]) -> None:
        best = self.root / "best"
        tmp = self.root / "best.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for f in (ADAPTER, "adapter_config.json", "trainer_state.json"):
            shutil.copyfile(src / f, tmp / f)
        (tmp / "best_metrics.json").write_text(json.dumps(metrics, indent=1))
        shutil.rmtree(best, ignore_errors=True)
        os.replace(tmp, best)
        if self.uploader is not None:
            files = {f"{self.prefix}/best/{f.name}": f for f in best.iterdir()}
            self.uploader.submit(files, f"best adapter ({metrics})")  # always queued

    def _upload(self, name: str) -> None:
        if self.uploader is None:
            return
        folder = self.root / name
        latest = self.root / f"LATEST.{name}"  # snapshot: LATEST may move before upload runs
        latest.write_text(name, encoding="utf-8")
        files = {f"{self.prefix}/{name}/{f.name}": f for f in folder.iterdir()}
        files[f"{self.prefix}/LATEST"] = latest
        for rel, path in self.extra_files.items():
            if path.exists():
                snap = self.root / f"{path.name}.{name}"
                shutil.copyfile(path, snap)
                files[rel] = snap

        def done() -> None:
            self.last_uploaded = name
            self._uploading = None
            for snap in [
                latest,
                *(self.root / f"{p.name}.{name}" for p in self.extra_files.values()),
            ]:
                snap.unlink(missing_ok=True)
            self._prune_remote()

        if self.uploader.try_submit(files, f"checkpoint {name}", on_done=done):
            self._uploading = name
        else:
            latest.unlink(missing_ok=True)
            log.info("uploader busy: %s stays local for now", name)

    def _prune_local(self) -> None:
        steps = sorted(p for p in self.root.iterdir() if p.is_dir() and STEP_RE.match(p.name))
        for old in steps[: -self.keep_local]:
            if old.name != self._uploading:
                shutil.rmtree(old, ignore_errors=True)

    def _prune_remote(self) -> None:
        assert self.mirror is not None
        names = sorted({p.split("/")[1] for p in self.mirror.list(f"{self.prefix}/step-")})
        old = names[: -self.keep_remote]
        if old:
            self.mirror.delete([f"{self.prefix}/{n}/" for n in old], f"prune {old}")

    def finalize(self) -> None:
        """Session end: wait for uploads; make sure the newest checkpoint reached the mirror."""
        if self.uploader is None:
            return
        self.uploader.wait()
        if self.last_saved and self.last_saved != self.last_uploaded:
            self._upload(self.last_saved)
            self.uploader.wait()
        if self.uploader.errors:
            log.error("checkpoint uploads failed: %s", self.uploader.errors)

    # ------------------------------------------------------------------------ reading
    def latest_local(self) -> Path | None:
        latest = self.root / "LATEST"
        if latest.exists():
            path = self.root / latest.read_text(encoding="utf-8").strip()
            if (path / "trainer_state.json").exists():
                return path
        steps = sorted(p for p in self.root.iterdir() if p.is_dir() and STEP_RE.match(p.name))
        return steps[-1] if steps else None

    def fetch_remote(self) -> Path | None:
        """Download the mirror's ``LATEST`` checkpoint into ``root`` (fresh Kaggle session)."""
        if self.mirror is None:
            return None
        try:
            name = (
                self.mirror.download(f"{self.prefix}/LATEST", self.root / "remote")
                .read_text()
                .strip()
            )
        except FileNotFoundError:
            return None
        files = [p for p in self.mirror.list(f"{self.prefix}/{name}/")]
        for rel in files:
            self.mirror.download(rel, self.root / "remote")
        src = self.root / "remote" / self.prefix / name
        dst = self.root / name
        if not dst.exists():
            shutil.copytree(src, dst)
        for rel, path in self.extra_files.items():  # e.g. continue the same train log
            try:
                got = self.mirror.download(rel, self.root / "remote")
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(got, path)
            except FileNotFoundError:
                pass
        self.last_saved = self.last_uploaded = name
        return dst

    def resume_path(self) -> Path | None:
        return self.latest_local() or self.fetch_remote()

    @staticmethod
    def load(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
        from safetensors.torch import load_file

        return {
            "adapter": load_file(str(path / ADAPTER)),
            "optimizer": torch.load(path / "optimizer.pt", map_location=device, weights_only=False),
            "scaler": torch.load(path / "scaler.pt", map_location="cpu", weights_only=False),
            "rng": torch.load(path / "rng.pt", map_location="cpu", weights_only=False),
            "state": json.loads((path / "trainer_state.json").read_text()),
        }
