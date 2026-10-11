"""Benchmark loaders, addressable by name from the CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from satquery.eval.datasets.base import (
    BenchmarkDataset,
    BenchmarkSample,
    DatasetNotFoundError,
    ImageIndex,
)
from satquery.eval.datasets.bigearthnet_txt import BigEarthNetTxt
from satquery.eval.datasets.rsvqa import CDVQA, RSVQALR
from satquery.eval.datasets.vrsbench import VRSBench
from satquery.types import TaskType

DATASETS: dict[str, Callable[..., BenchmarkDataset]] = {
    "rsvqa-lr": lambda root, split="test", **kw: RSVQALR(root, split),
    "cdvqa": lambda root, split="test", **kw: CDVQA(root, split, **kw),
    "vrsbench-vqa": lambda root, split="eval", **kw: VRSBench(root, TaskType.VQA, split),
    "vrsbench-caption": lambda root, split="eval", **kw: VRSBench(root, TaskType.CAPTION, split),
    "vrsbench-grounding": lambda root, split="eval", **kw: VRSBench(
        root, TaskType.GROUNDING, split
    ),
    "bigearthnet-txt": lambda root, split="bench", **kw: BigEarthNetTxt(root, split, **kw),
}

DEFAULT_ROOTS: dict[str, str] = {
    "rsvqa-lr": "datasets/rsvqa-lr",
    "cdvqa": "datasets/cdvqa",
    "vrsbench-vqa": "datasets/vrsbench",
    "vrsbench-caption": "datasets/vrsbench",
    "vrsbench-grounding": "datasets/vrsbench",
    "bigearthnet-txt": "datasets/bigearthnet-txt",
}


def load_dataset(name: str, root: Path | None = None, **kwargs) -> BenchmarkDataset:
    if name not in DATASETS:
        raise KeyError(f"unknown benchmark '{name}'; choose from {sorted(DATASETS)}")
    return DATASETS[name](Path(root or DEFAULT_ROOTS[name]), **kwargs)


__all__ = [
    "DATASETS",
    "DEFAULT_ROOTS",
    "BenchmarkDataset",
    "BenchmarkSample",
    "DatasetNotFoundError",
    "ImageIndex",
    "load_dataset",
]
