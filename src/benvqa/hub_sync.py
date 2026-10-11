"""Remote storage for shards and checkpoints: Hugging Face Hub or a local directory.

``Mirror`` is the small interface every component uses (shard builder uploads, the
streaming loader downloads, checkpoints mirror). ``LocalMirror`` backs tests and local
smoke runs. ``FlakyMirror`` injects failures for the network-robustness tests.
``AsyncUploader`` runs uploads on a background thread so training never blocks on them.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


class TransientError(RuntimeError):
    """A failure worth retrying (network hiccup, injected fault)."""


def retry[T](
    fn: Callable[[], T],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    what: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with exponential backoff + jitter. Re-raises the last error."""
    for i in range(attempts):
        try:
            return fn()
        except (KeyboardInterrupt, FileNotFoundError):
            raise
        except Exception as exc:
            if i == attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2**i) * (0.5 + random.random() / 2)
            log.warning(
                "%s failed (%s: %s); retry %d/%d in %.1fs",
                what,
                type(exc).__name__,
                exc,
                i + 1,
                attempts - 1,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")


class Mirror(Protocol):
    def upload(self, files: dict[str, Path], message: str) -> None:
        """Upload ``{path_in_repo: local_path}`` as one atomic commit."""

    def download(self, path: str, local_dir: Path) -> Path:
        """Fetch ``path`` to ``local_dir/path`` and return it. Missing -> FileNotFoundError."""

    def list(self, prefix: str = "") -> list[str]:
        """Repo paths under ``prefix``."""

    def delete(self, paths: Iterable[str], message: str) -> None: ...

    def describe(self) -> str: ...


class LocalMirror:
    """A directory standing in for a Hub repo (tests, local dry runs, smoke training)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def upload(self, files: dict[str, Path], message: str) -> None:
        for rel, src in files.items():
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".uploading")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)

    def download(self, path: str, local_dir: Path) -> Path:
        src = self.root / path
        if not src.exists():
            raise FileNotFoundError(f"{path} not in {self.root}")
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        return dst

    def list(self, prefix: str = "") -> list[str]:
        """Repo paths starting with ``prefix`` (string prefix, like the Hub listing)."""
        top = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
        base = self.root / top
        if not base.exists():
            return []
        files = (p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())
        return sorted(f for f in files if f.startswith(prefix) and not f.endswith(".uploading"))

    def delete(self, paths: Iterable[str], message: str) -> None:
        for rel in paths:
            target = self.root / rel
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    def describe(self) -> str:
        return f"local:{self.root}"


class HubMirror:
    """A Hugging Face Hub repo. Downloads go through ``hf_hub_download`` into ``local_dir``
    (not the shared cache), so the caller controls disk usage and can delete consumed files.
    """

    def __init__(
        self,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str | None = None,
        token: str | None = None,
        create: bool = False,
        private: bool = False,
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id, self.repo_type, self.revision = repo_id, repo_type, revision
        self.api = HfApi(token=token)
        if create:
            self.api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    def upload(self, files: dict[str, Path], message: str) -> None:
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
        retry(
            lambda: self.api.create_commit(
                self.repo_id, ops, commit_message=message, repo_type=self.repo_type
            ),
            what=f"upload {list(files)[:3]}",
        )

    def download(self, path: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        def fetch() -> Path:
            try:
                return Path(
                    hf_hub_download(
                        self.repo_id,
                        path,
                        repo_type=self.repo_type,
                        revision=self.revision,
                        local_dir=str(local_dir),
                        token=self.api.token,
                    )
                )
            except EntryNotFoundError as exc:
                raise FileNotFoundError(path) from exc

        return retry(fetch, what=f"download {path}")

    def list(self, prefix: str = "") -> list[str]:
        files = retry(
            lambda: self.api.list_repo_files(
                self.repo_id, repo_type=self.repo_type, revision=self.revision
            ),
            what="list repo",
        )
        return sorted(f for f in files if f.startswith(prefix))

    def delete(self, paths: Iterable[str], message: str) -> None:
        from huggingface_hub import CommitOperationDelete

        ops = [CommitOperationDelete(path_in_repo=p, is_folder=p.endswith("/")) for p in paths]
        if ops:
            retry(
                lambda: self.api.create_commit(
                    self.repo_id, ops, commit_message=message, repo_type=self.repo_type
                ),
                what="delete",
            )

    def head_revision(self) -> str:
        info = self.api.repo_info(self.repo_id, repo_type=self.repo_type, revision=self.revision)
        return info.sha

    def describe(self) -> str:
        return f"hub:{self.repo_type}/{self.repo_id}@{self.revision or 'main'}"


@dataclass
class FlakyMirror:
    """Wraps a mirror; each download/upload fails with probability ``p`` (deterministic)."""

    inner: Mirror
    p: float = 0.3
    seed: int = 0
    failures: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._lock = threading.Lock()

    def _maybe_fail(self, what: str) -> None:
        with self._lock:
            fail = self._rng.random() < self.p
            if fail:
                self.failures += 1
        if fail:
            raise TransientError(f"injected failure: {what}")

    def upload(self, files: dict[str, Path], message: str) -> None:
        retry(
            lambda: (self._maybe_fail("upload"), self.inner.upload(files, message))[1],
            base_delay=0.001,
            attempts=20,
            what="flaky upload",
        )

    def download(self, path: str, local_dir: Path) -> Path:
        def go() -> Path:
            self._maybe_fail(f"download {path}")
            return self.inner.download(path, local_dir)

        return retry(go, base_delay=0.001, attempts=20, what="flaky download")

    def list(self, prefix: str = "") -> list[str]:
        return self.inner.list(prefix)

    def delete(self, paths: Iterable[str], message: str) -> None:
        self.inner.delete(paths, message)

    def describe(self) -> str:
        return f"flaky({self.inner.describe()})"


def make_mirror(spec: str, **kwargs) -> Mirror:
    """``local:/path``, ``hub:<repo_id>`` (or a bare ``user/repo``), or ``flaky:<p>:<spec>``
    (fault injection for tests)."""
    if spec.startswith("flaky:"):
        _, p, rest = spec.split(":", 2)
        return FlakyMirror(make_mirror(rest, **kwargs), p=float(p))
    if spec.startswith("local:"):
        return LocalMirror(Path(spec.removeprefix("local:")))
    return HubMirror(spec.removeprefix("hub:"), **kwargs)


class AsyncUploader:
    """One background thread draining an upload queue.

    ``try_submit`` refuses new work while an upload is in flight, matching the checkpoint
    policy "if the uploader is busy the new checkpoint stays local-only". ``submit`` always
    enqueues (shard builder). ``wait`` blocks until the queue is drained.
    """

    def __init__(self, mirror: Mirror, name: str = "uploader") -> None:
        self.mirror = mirror
        self._q: queue.Queue[tuple[dict[str, Path], str, Callable[[], None] | None] | None] = (
            queue.Queue()
        )
        self._busy = threading.Event()
        self.errors: list[str] = []
        self.completed = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            files, message, on_done = item
            try:
                self.mirror.upload(files, message)
                self.completed += 1
                if on_done is not None:
                    on_done()
            except Exception as exc:  # recorded, never raised into the training loop
                log.error("upload failed for good: %s: %s", type(exc).__name__, exc)
                self.errors.append(f"{message}: {type(exc).__name__}: {exc}")
            finally:
                if self._q.unfinished_tasks <= 1:
                    self._busy.clear()
                self._q.task_done()

    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(
        self, files: dict[str, Path], message: str, on_done: Callable[[], None] | None = None
    ) -> None:
        self._busy.set()
        self._q.put((files, message, on_done))

    def try_submit(
        self, files: dict[str, Path], message: str, on_done: Callable[[], None] | None = None
    ) -> bool:
        if self.busy():
            return False
        self.submit(files, message, on_done)
        return True

    def wait(self) -> None:
        self._q.join()

    def close(self) -> None:
        self.wait()
        self._q.put(None)
        self._thread.join(timeout=60)
