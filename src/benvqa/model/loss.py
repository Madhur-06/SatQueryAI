"""Label-position-only LM head + turn-level normalised cross-entropy.

The tied LM head is a 2560 x 248,320 matmul, so full-vocab logits for a 600-token
conversation cost ~0.6 GB in fp32. Only the hidden states whose *next* token is a label
(~2 per QA turn) go through the head here.

Loss = sum_t w_t * mean_{tokens in turn t} CE / W, where ``W`` is the turn-weight sum of
the whole optimizer step (all micro-batches, all ranks). The trainer passes
``weight_norm = W / world_size`` because DDP averages gradients over ranks: summing over
micro-batches and averaging over ranks then gives exactly the global turn-weighted mean.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from torch import nn

from benvqa.data.vision import to_pixel_values


def turn_losses(ce: torch.Tensor, label_turn: torch.Tensor, n_turns: int) -> torch.Tensor:
    """Per-token CE -> mean CE per turn (``[n_turns]``)."""
    sums = torch.zeros(n_turns, dtype=ce.dtype, device=ce.device).index_add_(0, label_turn, ce)
    counts = torch.bincount(label_turn, minlength=n_turns).clamp_min(1).to(ce.dtype)
    return sums / counts


class LabelPositionLM(nn.Module):
    """Wraps the (PEFT) model so DDP sees one module whose forward returns the loss."""

    def __init__(self, model: nn.Module, base: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._base = [base]  # not a submodule (it is inside ``model`` already)

    @property
    def base(self) -> Any:
        return self._base[0]

    def hidden_states(self, batch: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        out = self.base.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            pixel_values=to_pixel_values(batch["pixel_tiles"], dtype),
            image_grid_thw=batch["image_grid_thw"],
            use_cache=False,
        )
        return out.last_hidden_state

    def forward(
        self, batch: dict[str, torch.Tensor], weight_norm: torch.Tensor | float
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dtype = self.base.lm_head.weight.dtype
        h = self.hidden_states(batch, dtype)
        h = h.reshape(-1, h.shape[-1]).index_select(0, batch["label_pos"])
        logits = self.base.lm_head(h).float()
        ce = F.cross_entropy(logits, batch["label_ids"], reduction="none")
        per_turn = turn_losses(ce, batch["label_turn"], batch["turn_weight"].numel())
        loss = (batch["turn_weight"] * per_turn).sum() / weight_norm
        # first label token of each turn: argmax accuracy (yes/no, letters) for logging
        first = torch.ones_like(batch["label_turn"], dtype=torch.bool)
        first[1:] = batch["label_turn"][1:] != batch["label_turn"][:-1]
        correct = (logits[first].argmax(-1) == batch["label_ids"][first]).float()
        return loss, {"turn_loss": per_turn.detach(), "turn_correct": correct.detach()}


def reference_loss(
    base: nn.Module, batch: dict[str, torch.Tensor], weight_norm: float
) -> torch.Tensor:
    """Same objective computed the naive way (full logits at every position); tests only.
    ``base`` is the ``*ForConditionalGeneration`` module."""
    out = base(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        mm_token_type_ids=batch["mm_token_type_ids"],
        pixel_values=to_pixel_values(batch["pixel_tiles"], base.lm_head.weight.dtype),
        image_grid_thw=batch["image_grid_thw"],
        use_cache=False,
    )
    logits = out.logits.float()  # [B, L, V]
    B, L, V = logits.shape
    labels = torch.full((B * L,), -100, dtype=torch.long, device=logits.device)
    turn = torch.full((B * L,), -1, dtype=torch.long, device=logits.device)
    labels[batch["label_pos"]] = batch["label_ids"]
    turn[batch["label_pos"]] = batch["label_turn"]
    ce = F.cross_entropy(logits.reshape(-1, V), labels, reduction="none", ignore_index=-100)
    mask = labels != -100
    per_turn = turn_losses(ce[mask], turn[mask], batch["turn_weight"].numel())
    return (batch["turn_weight"] * per_turn).sum() / weight_norm
