"""JSONL training log (+ optional Weights & Biases when ``WANDB_API_KEY`` is set)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [r{rank}] %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class JsonlLogger:
    def __init__(
        self, path: Path, config: dict[str, Any] | None = None, run_name: str = ""
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wandb = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self._wandb = wandb.init(
                    project="benvqa",
                    name=run_name or None,
                    config=config,
                    resume="allow",
                    id=run_name or None,
                )
            except Exception as exc:  # optional: never break training over it
                log.warning("wandb disabled: %s", exc)

    def write(self, record: dict[str, Any]) -> None:
        record = {"time": round(time.time(), 1), **record}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_jsonable) + "\n")
        if self._wandb is not None:
            flat = {k: v for k, v in record.items() if isinstance(v, int | float)}
            for k, v in record.items():
                if isinstance(v, dict):
                    flat.update(
                        {f"{k}/{kk}": vv for kk, vv in v.items() if isinstance(vv, int | float)}
                    )
            self._wandb.log(flat, step=record.get("step"))


def _jsonable(x: Any) -> Any:
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def gpu_memory_gb() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "alloc": round(torch.cuda.memory_allocated() / 2**30, 2),
            "peak": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.memory_reserved() / 2**30, 2),
        }
    except ImportError:  # pragma: no cover
        return {}
