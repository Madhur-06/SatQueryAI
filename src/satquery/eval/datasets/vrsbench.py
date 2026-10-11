"""VRSBench evaluation splits (VQA, detailed captioning, referring-expression grounding).

Files (verified on the Hugging Face dataset page): ``VRSBench_EVAL_vqa.json``,
``VRSBench_EVAL_Cap.json`` and ``VRSBench_EVAL_referring.json``, each a JSON list of
``{image_id, question, ground_truth, type, question_id, dataset}``. Referring
``ground_truth`` is ``"{<x1><y1><x2><y2>}"`` on a 0-100 scale. Images come from
``Images_val.zip``; they are located by file name anywhere under the root, so the
archive's internal folder layout does not matter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from satquery.eval.datasets.base import BenchmarkDataset, BenchmarkSample, ImageIndex
from satquery.types import Modality, TaskType

_FILES = {
    TaskType.VQA: "VRSBench_EVAL_vqa.json",
    TaskType.CAPTION: "VRSBench_EVAL_Cap.json",
    TaskType.GROUNDING: "VRSBench_EVAL_referring.json",
}
_BOX = re.compile(r"<\s*(-?[\d.]+)\s*>")


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    values = [float(v) for v in _BOX.findall(text)]
    return (values[0], values[1], values[2], values[3]) if len(values) >= 4 else None


class VRSBench(BenchmarkDataset):
    tasks = frozenset(_FILES)
    modality = Modality.OPTICAL

    def __init__(self, root: Path, task: TaskType, split: str = "eval") -> None:
        super().__init__(root, split)
        if task not in _FILES:
            raise ValueError(f"VRSBench has no {task} split")
        self.task = task
        self.name = f"vrsbench-{task.value}"

    def samples(self) -> Iterator[BenchmarkSample]:
        path = self.root / _FILES[self.task]
        self.require(path)
        items = json.loads(path.read_text(encoding="utf-8"))
        index = ImageIndex(self.root)
        for i, item in enumerate(items):
            image = index.get(item["image_id"])
            if image is None:
                continue
            gt = str(item["ground_truth"])
            bbox = parse_box(gt) if self.task is TaskType.GROUNDING else None
            if self.task is TaskType.GROUNDING and bbox is None:
                continue
            yield BenchmarkSample(
                id=f"{self.name}:{item.get('question_id', i)}",
                task=self.task,
                images=[image],
                question=item.get("question"),
                question_type=item.get("type"),
                answers=[] if bbox else [gt],
                bbox=bbox,
                meta={"image_id": item["image_id"]},
            )
