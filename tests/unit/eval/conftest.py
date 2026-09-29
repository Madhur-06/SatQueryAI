"""Tiny synthetic copies of each benchmark's on-disk layout (no real data)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from satquery.testing.synthetic import SceneGeometry, write_array


def _rgb(seed: int, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)


def _write_rsvqa_style(root: Path, prefix: str, with_file_name: bool) -> None:
    questions = [
        {"id": 0, "img_id": 0, "type": "presence", "question": "Is there a road?",
         "answers_ids": [0], "active": True},
        {"id": 1, "img_id": 0, "type": "count", "question": "How many buildings are there?",
         "answers_ids": [1], "active": True},
        {"id": 2, "img_id": 1, "type": "presence", "question": "Is there water?",
         "answers_ids": [2], "active": True},
        {"id": 3, "img_id": 1, "type": "comp", "question": "Are there more roads than water?",
         "answers_ids": [3], "active": False},  # inactive: must be skipped
    ]  # fmt: skip
    answers = [
        {"id": 0, "question_id": 0, "answer": "yes", "active": True},
        {"id": 1, "question_id": 1, "answer": "0", "active": True},
        {"id": 2, "question_id": 2, "answer": "no", "active": True},
        {"id": 3, "question_id": 3, "answer": "yes", "active": True},
    ]
    images = [
        {"id": i, "questions_ids": [], "active": True, **({"file_name": f"{i:05d}.png"}
         if with_file_name else {})}
        for i in range(2)
    ]  # fmt: skip
    for kind, items in (("questions", questions), ("answers", answers), ("images", images)):
        (root / f"{prefix}_{kind}.json").write_text(json.dumps({kind: items}), encoding="utf-8")


@pytest.fixture
def rsvqa_root(tmp_path: Path) -> Path:
    root = tmp_path / "rsvqa-lr"
    (root / "Images_LR").mkdir(parents=True)
    _write_rsvqa_style(root, "LR_split_test", with_file_name=False)
    for i in range(2):
        data = np.moveaxis(_rgb(i), -1, 0)
        write_array(root / "Images_LR" / f"{i}.tif", data, SceneGeometry(size=64, crs=None))
    return root


@pytest.fixture
def cdvqa_root(tmp_path: Path) -> Path:
    root = tmp_path / "cdvqa"
    for folder in ("im1", "im2"):
        (root / folder).mkdir(parents=True)
        for i in range(2):
            Image.fromarray(_rgb(i + (10 if folder == "im2" else 0))).save(
                root / folder / f"{i:05d}.png"
            )
    _write_rsvqa_style(root, "Test", with_file_name=True)
    return root


@pytest.fixture
def vrsbench_root(tmp_path: Path) -> Path:
    root = tmp_path / "vrsbench"
    (root / "Images_val" / "nested").mkdir(parents=True)
    for name in ("P0001_0001.png", "P0002_0001.png"):
        Image.fromarray(_rgb(len(name))).save(root / "Images_val" / "nested" / name)
    vqa = [
        {"image_id": "P0001_0001.png", "question": "Is there a ship?", "ground_truth": "Yes",
         "dataset": "RSBench", "question_id": 0, "type": "object existence"},
        {"image_id": "P0002_0001.png", "question": "How many cars are there?",
         "ground_truth": "3", "dataset": "RSBench", "question_id": 1, "type": "object quantity"},
        {"image_id": "missing.png", "question": "?", "ground_truth": "no", "question_id": 2,
         "type": "x"},
    ]  # fmt: skip
    cap = [
        {"image_id": "P0001_0001.png", "question": "Describe the image in detail",
         "ground_truth": "A harbour with two ships docked near a pier.", "question_id": 0,
         "type": "caption"},
    ]  # fmt: skip
    ref = [
        {"image_id": "P0001_0001.png", "question": "The ship at the top left.",
         "ground_truth": "{<0><0><100><100>}", "question_id": 0, "type": "ref"},
        {"image_id": "P0002_0001.png", "question": "The small car on the right.",
         "ground_truth": "{<60><40><80><60>}", "question_id": 1, "type": "ref"},
    ]  # fmt: skip
    for name, items in (("vqa", vqa), ("Cap", cap), ("referring", ref)):
        (root / f"VRSBench_EVAL_{name}.json").write_text(json.dumps(items), encoding="utf-8")
    return root
