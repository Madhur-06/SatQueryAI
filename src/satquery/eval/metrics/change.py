"""Binary change-mask metrics: precision, recall, F1 and IoU (micro over all pixels, plus the
per-image mean)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def change_mask_scores(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """``pairs``: (predicted mask, ground-truth mask), boolean arrays of equal shape."""
    tp = fp = fn = 0
    per_image = []
    for pred, gt in pairs:
        p, g = np.asarray(pred, bool), np.asarray(gt, bool)
        if p.shape != g.shape:
            raise ValueError(f"shape mismatch {p.shape} vs {g.shape}")
        t, f_p, f_n = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        per_image.append(_prf(t, f_p, f_n))
    micro = _prf(tp, fp, fn)
    mean = {k: float(np.mean([m[k] for m in per_image])) if per_image else 0.0 for k in micro}
    return {"n": len(pairs), "micro": micro, "mean_per_image": mean}
