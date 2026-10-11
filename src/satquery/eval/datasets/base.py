"""Benchmark dataset interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from satquery.types import Modality, TaskType


class DatasetNotFoundError(FileNotFoundError):
    """Raised with human-readable instructions when benchmark files are missing."""


class BenchmarkSample(BaseModel):
    id: str
    task: TaskType
    images: list[Path]
    """One image, or (before, after) for change tasks."""
    question: str | None = None
    question_type: str | None = None
    answers: list[str] = Field(default_factory=list)
    """Accepted answers (VQA) or reference captions (captioning)."""
    bbox: tuple[float, float, float, float] | None = None
    """Ground-truth box for grounding, (x1, y1, x2, y2) on a 0-100 normalised scale."""
    change_mask: Path | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class BenchmarkDataset(ABC):
    name: str
    tasks: frozenset[TaskType]
    modality: Modality | None = None
    """Known modality of every image (skips heuristic detection on tiny benchmark tiles)."""

    def __init__(self, root: Path, split: str = "test") -> None:
        self.root = Path(root)
        self.split = split

    @abstractmethod
    def samples(self) -> Iterator[BenchmarkSample]:
        """All samples of the split, in a stable order."""

    def subset(self, n: int | None, seed: int = 0) -> list[BenchmarkSample]:
        """A deterministic random subset of ``n`` samples (all if ``n`` is None)."""
        items = list(self.samples())
        if n is None or n >= len(items):
            return items
        return random.Random(seed).sample(items, n)

    def require(self, *paths: Path) -> None:
        missing = [p for p in paths if not p.exists()]
        if missing:
            from satquery.eval.sources import SOURCES

            source = SOURCES.get(self.name.split(":")[0])
            where = f"\nSource: {source.homepage}\nFiles: {source.files}" if source else ""
            raise DatasetNotFoundError(
                f"{self.name}: missing {[str(p) for p in missing]} under {self.root}.{where}"
            )


class ImageIndex:
    """Find image files by name anywhere under a root (archives unpack to varying layouts)."""

    def __init__(self, root: Path, suffixes: tuple[str, ...] = (".png", ".jpg", ".tif")) -> None:
        self._by_name: dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in suffixes:
                self._by_name.setdefault(path.name, path)

    def get(self, name: str) -> Path | None:
        return self._by_name.get(name)
