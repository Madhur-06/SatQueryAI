"""Grounding metrics: Acc@IoU>=0.5 (the VRSBench referring metric) and mean IoU."""

from __future__ import annotations

from typing import Any

Box = tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def grounding_scores(pairs: list[tuple[Box | None, Box]], threshold: float = 0.5) -> dict[str, Any]:
    """``pairs``: (predicted box or None, ground-truth box), same coordinate frame."""
    ious = [box_iou(p, g) if p is not None else 0.0 for p, g in pairs]
    n = len(ious)
    return {
        "n": n,
        f"acc@{threshold}": sum(i >= threshold for i in ious) / n if n else 0.0,
        "mean_iou": sum(ious) / n if n else 0.0,
    }
