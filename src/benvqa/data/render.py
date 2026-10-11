"""Conversation -> token ids, labels and turn bookkeeping, built segment by segment.

Every assistant turn starts with exactly the non-thinking generation prefix the official
Qwen3.5 template emits for a single-turn prompt (``add_generation_prompt=True,
enable_thinking=False``): ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``. The template
itself would render *historical* assistant turns without the think block, so multi-turn
samples are assembled here from fixed segments rather than with ``apply_chat_template``.
Segment boundaries sit on special tokens or newlines, so segment-wise tokenization equals
whole-string tokenization (tested).

Labels cover the answer tokens and the closing ``<|im_end|>`` of each assistant turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benvqa.data.conversation import Conversation, Turn
from benvqa.data.vision import tokens_per_image
from benvqa.prompting import IMAGE_TAGS, SYSTEM_PROMPT

IGNORE = -100
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass
class Rendered:
    input_ids: np.ndarray  # int32 [L]
    labels: np.ndarray  # int32 [L], IGNORE outside answers
    label_turn: np.ndarray  # int16 [L], turn index for label positions, -1 elsewhere
    turn_weight: np.ndarray  # float32 [T]
    turn_category: list[str]
    turn_kind: list[str]
    composites: tuple[str, ...]
    n_dropped_turns: int = 0

    def __len__(self) -> int:
        return len(self.input_ids)


class Renderer:
    def __init__(self, tokenizer: Any, image_size: int = 256, max_len: int = 1024) -> None:
        self.tok = tokenizer
        self.max_len = max_len
        self.n_image_tokens = tokens_per_image(image_size)
        enc = self.encode
        self.sys = enc(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
        self.user_open = enc("<|im_start|>user\n")
        self.user_close = enc("<|im_end|>\n")
        self.asst_prefix = enc(ASSISTANT_PREFIX)
        self.im_end = self._single("<|im_end|>")
        self.newline = enc("\n")
        self.vision_start = self._single("<|vision_start|>")
        self.vision_end = self._single("<|vision_end|>")
        self.image_pad = self._single("<|image_pad|>")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._tag_cache: dict[tuple[str, bool], list[int]] = {}

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def _single(self, text: str) -> int:
        ids = self.encode(text)
        assert len(ids) == 1, f"{text!r} is not a single token: {ids}"
        return ids[0]

    def _tag(self, name: str, first: bool) -> list[int]:
        key = (name, first)
        if key not in self._tag_cache:
            self._tag_cache[key] = self.encode(("" if first else "\n") + IMAGE_TAGS[name])
        return self._tag_cache[key]

    def image_block(self, composites: tuple[str, ...]) -> list[int]:
        ids: list[int] = []
        for i, name in enumerate(composites):
            ids += self._tag(name, i == 0)
            ids += [self.vision_start] + [self.image_pad] * self.n_image_tokens + [self.vision_end]
        return ids

    def _user(self, question: str, composites: tuple[str, ...] | None) -> list[int]:
        if composites:
            return (
                self.user_open
                + self.image_block(composites)
                + self.encode("\n" + question)
                + self.user_close
            )
        return self.user_open + self.encode(question) + self.user_close

    def prompt_ids(self, question: str, composites: tuple[str, ...]) -> list[int]:
        """Single-turn inference prompt, ending with the assistant prefix (score-based eval)."""
        return self.sys + self._user(question, composites) + self.asst_prefix

    def render(self, conv: Conversation) -> Rendered:
        turns: list[Turn] = list(conv.turns)
        pieces = self._pieces(conv, turns)
        dropped = 0
        while sum(len(p[0]) for p in pieces) > self.max_len and len(turns) > 1:
            cap = [i for i, t in enumerate(turns) if t.kind == "caption"]
            turns.pop(cap[-1] if cap else len(turns) - 1)
            dropped += 1
            pieces = self._pieces(conv, turns)
        ids: list[int] = []
        labels: list[int] = []
        label_turn: list[int] = []
        for seg, is_label, t_idx in pieces:
            ids += seg
            labels += seg if is_label else [IGNORE] * len(seg)
            label_turn += [t_idx] * len(seg) if is_label else [-1] * len(seg)
        return Rendered(
            input_ids=np.asarray(ids, dtype=np.int32),
            labels=np.asarray(labels, dtype=np.int32),
            label_turn=np.asarray(label_turn, dtype=np.int16),
            turn_weight=np.asarray([t.weight for t in turns], dtype=np.float32),
            turn_category=[t.category for t in turns],
            turn_kind=[t.kind for t in turns],
            composites=conv.composites,
            n_dropped_turns=dropped,
        )

    def _pieces(self, conv: Conversation, turns: list[Turn]) -> list[tuple[list[int], bool, int]]:
        pieces: list[tuple[list[int], bool, int]] = [(self.sys, False, -1)]
        for i, t in enumerate(turns):
            if i > 0:
                pieces.append((self.newline, False, -1))  # after the previous <|im_end|>
            pieces.append((self._user(t.question, conv.composites if i == 0 else None), False, -1))
            pieces.append((self.asst_prefix, False, -1))
            pieces.append(([*self.encode(t.answer), self.im_end], True, i))
        return pieces
