"""One image visit -> one multi-turn, leakage-safe conversation (Stage 1).

Rules (Phase 2.1 brief section 6.2, with the M1 adjustment for unavailable items):

* Questions are grouped into leakage groups; **at most one item per group**:
  presence / area / count / adjacency {binary-yes, binary-no, MCQ}, relative_pos {MCQ},
  geo {country MCQ, climate MCQ} (correlated, one of them), season {MCQ}.
* presence/area/count/adjacency pick binary-yes / binary-no / MCQ with *absolute*
  probabilities (default 0.25 / 0.25 / 0.5). If the drawn item does not exist for this image
  the group is skipped. Renormalising would make adjacency binaries ~2:1 `no` (M1).
* relative_pos is always included when present (rare category).
* Class-mentioning questions (area, count, adjacency, relative_pos) reveal class presence,
  so a presence question always comes before all of them. With ``p_drop_presence``
  presence is omitted and the rest is shuffled freely.
* The caption (it contains almost every answer) is added with ``p_caption``, always last.
* MCQ options are shuffled per visit (round-trip checked; see ``mcq.py``).
* Modality dropout picks which composites accompany the conversation.

Randomness is a pure function of ``(seed, patch_id, visit)``. Kind choices cycle across
visits (a per-image weighted order), so a re-visited image gets items it has not seen yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from benvqa.data.mcq import shuffle_mcq

Kind = Literal["binary", "mcq", "caption"]

GROUP_OF = {
    "presence": "presence",
    "area": "area",
    "count": "count",
    "adjacency": "adjacency",
    "relative pos": "relative_pos",
    "country": "geo",
    "climate zone": "geo",
    "season": "season",
}
TRIPLE_GROUPS = ("presence", "area", "count", "adjacency")
CLASS_MENTIONING = frozenset({"area", "count", "adjacency", "relative_pos"})
PICK_KINDS = ("binary_yes", "binary_no", "mcq")


class ConversationConfig(BaseModel):
    group_pick_probs: dict[str, float] = Field(
        default_factory=lambda: {"binary_yes": 0.25, "binary_no": 0.25, "mcq": 0.5}
    )
    p_drop_presence: float = Field(0.3, ge=0, le=1)
    p_caption: float = Field(0.2, ge=0, le=1)
    caption_turn_weight: float = Field(1.0, ge=0)
    shuffle_mcq_options: bool = True
    always_relative_pos: bool = True
    composites: list[str] = Field(default_factory=lambda: ["rgb", "nir_swir", "sar"])
    modality_dropout: dict[str, float] = Field(
        default_factory=lambda: {
            "all": 0.60,
            "rgb+nir_swir": 0.15,
            "rgb+sar": 0.10,
            "rgb": 0.10,
            "sar": 0.05,
        }
    )

    @field_validator("group_pick_probs")
    @classmethod
    def _kinds(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != set(PICK_KINDS) or abs(sum(v.values()) - 1) > 1e-6:
            raise ValueError(f"group_pick_probs needs {PICK_KINDS} summing to 1, got {v}")
        return v

    @model_validator(mode="after")
    def _dropout(self) -> ConversationConfig:
        total = sum(self.modality_dropout.values())
        if abs(total - 1) > 1e-6:
            raise ValueError(f"modality_dropout must sum to 1, got {total}")
        for key in self.modality_dropout:
            if key != "all" and not set(key.split("+")) <= set(self.composites):
                raise ValueError(f"modality set {key!r} uses composites not in {self.composites}")
        return self

    def modality_set(self, key: str) -> tuple[str, ...]:
        names = self.composites if key == "all" else key.split("+")
        return tuple(c for c in self.composites if c in names)  # canonical order


@dataclass(frozen=True)
class QAItem:
    id: int
    type: str
    category: str
    input: str
    output: str

    @property
    def group(self) -> str | None:
        return GROUP_OF.get(self.category)

    @property
    def pick_kind(self) -> str:
        if self.type == "binary":
            return f"binary_{self.output}"
        return self.type  # mcq / captioning


@dataclass
class Turn:
    question: str
    answer: str
    kind: Kind
    category: str
    weight: float
    qa_id: int
    group: str | None = None
    shuffled: bool = False


@dataclass
class Conversation:
    patch_id: str
    visit: int
    composites: tuple[str, ...]
    turns: list[Turn] = field(default_factory=list)


def parse_qa(qa_json: str) -> list[QAItem]:
    return [QAItem(**d) for d in json.loads(qa_json)]


def _seed_words(*parts: object) -> list[int]:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=16).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


def rng_for(seed: int, patch_id: str, visit: int, tag: str) -> np.random.Generator:
    return np.random.default_rng(_seed_words(seed, patch_id, visit, tag))


def weighted_order(rng: np.random.Generator, items: list[str], weights: list[float]) -> list[str]:
    """Sample without replacement proportional to ``weights`` (Gumbel top-k)."""
    keys = np.log(np.asarray(weights, dtype=np.float64)) + rng.gumbel(size=len(items))
    return [items[i] for i in np.argsort(-keys)]


def group_items(items: list[QAItem]) -> tuple[dict[str, dict[str, list[QAItem]]], list[QAItem]]:
    groups: dict[str, dict[str, list[QAItem]]] = {}
    captions: list[QAItem] = []
    for it in items:
        if it.type == "captioning":
            captions.append(it)
        elif it.group is not None and it.type in ("binary", "mcq"):
            groups.setdefault(it.group, {}).setdefault(it.pick_kind, []).append(it)
    return groups, captions


def _cycled(rng: np.random.Generator, items: list[QAItem], k: int) -> QAItem:
    order = rng.permutation(len(items))
    return items[int(order[k % len(items)])]


def _turn(item: QAItem, weight: float = 1.0) -> Turn:
    kind: Kind = "caption" if item.type == "captioning" else item.type  # type: ignore[assignment]
    return Turn(item.input, item.output, kind, item.category, weight, item.id, item.group)


def build_conversation(
    items: list[QAItem], patch_id: str, visit: int, cfg: ConversationConfig, seed: int
) -> Conversation:
    groups, captions = group_items(items)
    rng = rng_for(seed, patch_id, visit, "conv")
    cyc = rng_for(seed, patch_id, 0, "cycle")  # visit-independent: defines the cycle order
    probs = cfg.group_pick_probs
    chosen: dict[str, QAItem] = {}

    for g in TRIPLE_GROUPS:
        order = weighted_order(cyc, list(PICK_KINDS), [probs[k] for k in PICK_KINDS])
        if g not in groups:
            continue
        kind = order[visit % len(order)]
        options = groups[g].get(kind)
        if options:  # absolute probabilities: an unavailable pick skips the group
            chosen[g] = _cycled(cyc, options, visit // len(order))
    for g in ("relative_pos", "season"):
        opts = [it for its in groups.get(g, {}).values() for it in its]
        if opts and (g != "relative_pos" or cfg.always_relative_pos):
            chosen[g] = _cycled(cyc, opts, visit)
    geo = sorted((it for its in groups.get("geo", {}).values() for it in its), key=lambda i: i.id)
    if geo:
        chosen["geo"] = _cycled(cyc, geo, visit)

    presence = chosen.pop("presence", None)
    if presence is not None and rng.random() < cfg.p_drop_presence:
        presence = None
    others = [chosen[g] for g in sorted(chosen)]
    others = [others[i] for i in rng.permutation(len(others))]
    if presence is not None:
        first_cm = next(
            (i for i, it in enumerate(others) if it.group in CLASS_MENTIONING), len(others)
        )
        others.insert(int(rng.integers(0, first_cm + 1)), presence)
    turns = [_turn(it) for it in others]

    if captions and rng.random() < cfg.p_caption:
        turns.append(_turn(_cycled(cyc, captions, visit), cfg.caption_turn_weight))

    if cfg.shuffle_mcq_options:
        for t in turns:
            if t.kind == "mcq":
                t.question, t.answer, t.shuffled = shuffle_mcq(t.question, t.answer, rng)

    keys = list(cfg.modality_dropout)
    key = keys[int(rng.choice(len(keys), p=[cfg.modality_dropout[k] for k in keys]))]
    return Conversation(patch_id, visit, cfg.modality_set(key), turns)


def single_turn(item: QAItem, composites: tuple[str, ...], patch_id: str = "") -> Conversation:
    """Inference/eval format: one question, inputs verbatim (no option shuffling)."""
    return Conversation(patch_id, 0, composites, [_turn(item)])


def check_leakage_rules(conv: Conversation) -> list[str]:
    """Violations of the leakage rules (empty list = ok). Used by tests and debug asserts."""
    problems = []
    groups = [t.group for t in conv.turns if t.kind != "caption"]
    if len(groups) != len(set(groups)):
        problems.append(f"repeated group: {groups}")
    kinds = [t.kind for t in conv.turns]
    if "caption" in kinds and kinds.index("caption") != len(kinds) - 1:
        problems.append("caption is not the last turn")
    if kinds.count("caption") > 1:
        problems.append("more than one caption")
    if "presence" in groups:
        p = groups.index("presence")
        if any(g in CLASS_MENTIONING for g in groups[:p]):
            problems.append("class-mentioning question before presence")
    cats = {t.category for t in conv.turns}
    if {"country", "climate zone"} <= cats:
        problems.append("country and climate zone together")
    return problems
