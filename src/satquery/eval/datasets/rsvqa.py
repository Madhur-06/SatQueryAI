"""RSVQA-style benchmarks: RSVQA-LR and CDVQA share one annotation schema.

Schema (as read by the RSVQA authors' ``VQA_model/VQALoader.py``, and identical in the
CDVQA repository): three JSON files with top-level keys ``questions``, ``answers`` and
``images``. Questions carry ``id, img_id, type, question, answers_ids, active``;
answers carry ``id, question_id, answer, active``; images carry ``id, questions_ids,
active`` (CDVQA adds ``file_name``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from satquery.eval.datasets.base import BenchmarkDataset, BenchmarkSample
from satquery.types import Modality, TaskType


def _load(path: Path, key: str) -> dict[int, dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))[key]
    return {int(item.get("id", i)): item for i, item in enumerate(items)}


class _RSVQAStyle(BenchmarkDataset):
    task: TaskType

    def _files(self) -> tuple[Path, Path, Path]:
        raise NotImplementedError

    def _image_paths(self, image: dict[str, Any]) -> list[Path]:
        raise NotImplementedError

    def samples(self) -> Iterator[BenchmarkSample]:
        q_path, a_path, i_path = self._files()
        self.require(q_path, a_path, i_path)
        questions = _load(q_path, "questions")
        answers = _load(a_path, "answers")
        images = _load(i_path, "images")
        for qid in sorted(questions):
            q = questions[qid]
            if not q.get("active", True):
                continue
            image = images.get(int(q["img_id"]))
            if image is None or not image.get("active", True):
                continue
            accepted = [
                str(answers[int(a)]["answer"])
                for a in q.get("answers_ids", [])
                if int(a) in answers and answers[int(a)].get("active", True)
            ]
            if not accepted:
                continue
            yield BenchmarkSample(
                id=f"{self.name}:{self.split}:{qid}",
                task=self.task,
                images=self._image_paths(image),
                question=q["question"],
                question_type=q.get("type"),
                answers=accepted,
                meta={"img_id": q["img_id"]},
            )


class RSVQALR(_RSVQAStyle):
    """RSVQA-LR: ``<root>/LR_split_<split>_{questions,answers,images}.json`` and
    ``<root>/Images_LR/<img_id>.tif`` (the layout of the Zenodo record's files)."""

    name = "rsvqa-lr"
    tasks = frozenset({TaskType.VQA})
    modality = Modality.OPTICAL
    task = TaskType.VQA

    def _files(self) -> tuple[Path, Path, Path]:
        return tuple(  # type: ignore[return-value]
            self.root / f"LR_split_{self.split}_{kind}.json"
            for kind in ("questions", "answers", "images")
        )

    def _image_paths(self, image: dict[str, Any]) -> list[Path]:
        return [self.root / "Images_LR" / f"{image['id']}.tif"]


class CDVQA(_RSVQAStyle):
    """CDVQA: ``<root>/<Split>_{questions,answers,images}.json``; image pairs looked up as
    ``<root>/<before_dir>/<file_name>`` and ``<root>/<after_dir>/<file_name>``.

    The repository ships annotations only; images are the SECOND train set (2,968 pairs,
    ``im1`` = date 1, ``im2`` = date 2), fetched by ``satquery-eval --download cdvqa``.
    Verified 2026-09-29: every image referenced by the test and test2 splits resolves.
    """

    name = "cdvqa"
    tasks = frozenset({TaskType.CHANGE_VQA})
    modality = Modality.OPTICAL
    task = TaskType.CHANGE_VQA

    def __init__(
        self, root: Path, split: str = "test", before_dir: str = "im1", after_dir: str = "im2"
    ) -> None:
        super().__init__(root, split)
        self.before_dir, self.after_dir = before_dir, after_dir

    def _files(self) -> tuple[Path, Path, Path]:
        split = self.split[:1].upper() + self.split[1:]  # test -> Test, test2 -> Test2
        return tuple(  # type: ignore[return-value]
            self.root / f"{split}_{kind}.json" for kind in ("questions", "answers", "images")
        )

    def _image_paths(self, image: dict[str, Any]) -> list[Path]:
        name = image["file_name"]
        return [self.root / self.before_dir / name, self.root / self.after_dir / name]
