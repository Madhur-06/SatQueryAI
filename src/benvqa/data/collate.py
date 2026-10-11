"""Rendered conversations + image tiles -> one right-padded micro-batch of torch tensors.

Only label positions are materialised for the loss: ``label_pos`` holds flat indices
``b * L + p`` of the positions whose *next* token is a label, so the model applies the
LM head to those hidden states only (``benvqa.model.loss``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benvqa.data.render import IGNORE, Rendered

CATEGORIES = (
    "presence",
    "area",
    "count",
    "adjacency",
    "relative pos",
    "country",
    "season",
    "climate zone",
    "captioning",
)
KINDS = ("binary", "mcq", "caption")


@dataclass
class Item:
    rendered: Rendered
    tiles: list[np.ndarray]
    """uint8 ``[n_patches, 3, 16, 16]`` per image, in prompt order."""
    grids: list[tuple[int, int, int]]
    patch_id: str = ""


def padded_length(n: int, multiple: int) -> int:
    """Round up to a multiple: the fla/Triton kernels compile and autotune once per sequence
    length, so free-running lengths cost a recompile (and allocator growth) on most steps."""
    return -(-n // multiple) * multiple if multiple > 1 else n


def collate(
    items: list[Item],
    pad_id: int,
    image_pad_id: int,
    meta: dict[str, Any],
    pad_to_multiple: int = 1,
) -> dict:
    import torch

    B = len(items)
    L = padded_length(max(len(it.rendered) for it in items), pad_to_multiple)
    input_ids = np.full((B, L), pad_id, dtype=np.int64)
    attention = np.zeros((B, L), dtype=np.int64)
    label_pos, label_ids, label_turn = [], [], []
    turn_weight, turn_cat, turn_kind = [], [], []
    for b, it in enumerate(items):
        r = it.rendered
        n = len(r)
        input_ids[b, :n] = r.input_ids
        attention[b, :n] = 1
        pos = np.flatnonzero(r.labels != IGNORE)
        assert pos.size, "conversation without labels"
        assert pos[0] > 0
        label_pos.append(b * L + pos - 1)  # hidden state at p-1 predicts token p
        label_ids.append(r.labels[pos])
        label_turn.append(r.label_turn[pos].astype(np.int64) + len(turn_weight))
        turn_weight.extend(r.turn_weight.tolist())
        turn_cat.extend(CATEGORIES.index(c) for c in r.turn_category)
        turn_kind.extend(KINDS.index(k) for k in r.turn_kind)
    tiles = np.concatenate([t for it in items for t in it.tiles], axis=0)
    grids = np.asarray([g for it in items for g in it.grids], dtype=np.int64)
    n_tokens = int(attention.sum())
    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attention),
        "mm_token_type_ids": torch.from_numpy((input_ids == image_pad_id).astype(np.int32)),
        "pixel_tiles": torch.from_numpy(tiles),
        "image_grid_thw": torch.from_numpy(grids),
        "label_pos": torch.from_numpy(np.concatenate(label_pos).astype(np.int64)),
        "label_ids": torch.from_numpy(np.concatenate(label_ids).astype(np.int64)),
        "label_turn": torch.from_numpy(np.concatenate(label_turn)),
        "turn_weight": torch.tensor(turn_weight, dtype=torch.float32),
        "turn_category": torch.tensor(turn_cat, dtype=torch.int64),
        "turn_kind": torch.tensor(turn_kind, dtype=torch.int64),
        "meta": {
            **meta,
            "n_conversations": B,
            "n_turns": len(turn_weight),
            "n_qa": sum(k != KINDS.index("caption") for k in turn_kind),
            "n_tokens": n_tokens,
            "n_padded": B * L,
            "patch_ids": [it.patch_id for it in items],
        },
    }
