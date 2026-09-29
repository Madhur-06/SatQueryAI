"""VQA accuracy: exact match after normalisation, overall and per question type.

RSVQA papers report both overall accuracy (OA, over all questions) and average
accuracy (AA, the mean of per-type accuracies); both are returned.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip().lower()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def vqa_accuracy(records: list[tuple[Any, list[str], str | None]]) -> dict[str, Any]:
    """``records``: (prediction, accepted answers, question type). Missing predictions count
    as wrong."""
    per_type: dict[str, list[int]] = defaultdict(list)
    correct_total = 0
    for pred, answers, qtype in records:
        accepted = {normalize_answer(a) for a in answers}
        ok = int(pred is not None and normalize_answer(pred) in accepted)
        correct_total += ok
        per_type[qtype or "all"].append(ok)
    n = len(records)
    types = {t: {"accuracy": sum(v) / len(v), "n": len(v)} for t, v in sorted(per_type.items())}
    return {
        "n": n,
        "overall_accuracy": correct_total / n if n else 0.0,
        "average_accuracy": (
            sum(t["accuracy"] for t in types.values()) / len(types) if types else 0.0
        ),
        "per_type": types,
    }
