"""Stage 1: MCQ shuffling, leakage-safe conversations, determinism and statistics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from benvqa import testing as synth
from benvqa.data.conversation import (
    CLASS_MENTIONING,
    ConversationConfig,
    QAItem,
    build_conversation,
    check_leakage_rules,
)
from benvqa.data.mcq import parse_mcq, shuffle_mcq

CFG = ConversationConfig()


def _items(rows: list[dict]) -> list[QAItem]:
    out = []
    for r in rows:
        if r["type"] in ("binary", "mcq", "captioning"):
            cat = "captioning" if r["type"] == "captioning" else r["category"]
            out.append(QAItem(r["ID"], r["type"], cat, r["input"], r["output"]))
    return out


def _image(i: int, rng: np.random.Generator, **kw) -> tuple[str, list[QAItem]]:
    pid, s1 = synth.patch_names(i % 7, i)
    adj = kw.get("adjacency", bool(rng.random() < 0.79))
    rel = kw.get("relpos", bool(rng.random() < 0.2))
    return pid, _items(synth.synth_rows(pid, s1, "train", rng, adjacency=adj, relpos=rel))


def _bench_like(i: int, rng: np.random.Generator) -> tuple[str, list[QAItem]]:
    """Several items per kind, like the bench split."""
    pid, items = _image(i, rng, adjacency=True, relpos=True)
    extra = [
        QAItem(10_000 + k, it.type, it.category, it.input + f" (v{k})", it.output)
        for k in range(2)
        for it in items
        if it.type != "captioning"
    ]
    return pid, items + extra


# ---------------------------------------------------------------------------------- MCQ
@pytest.mark.parametrize(
    "text",
    [
        "Which country? a) Lithuania, b) Belgium, c) Austria, d) Luxembourg",
        "Select the climate zone: a) Cold, no dry season, warm summer, b) Temperate, dry "
        "summer, hot summer, c) Arid, steppe, cold, d) Cold, no dry season, cold summer",
        "Identify touching classes: a) Agro-forestry areas and Industrial or commercial units, "
        "b) Land principally occupied by agriculture, with significant areas of natural "
        "vegetation and Urban fabric, c) Pastures and Urban fabric, d) Marine waters and "
        "Transitional woodland, shrub",
    ],
)
def test_mcq_parse_roundtrip_and_shuffle_remap(text: str) -> None:
    mcq = parse_mcq(text)
    assert mcq is not None
    assert len(mcq.options) == 4
    assert mcq.render() == text
    for seed in range(20):
        for answer in "abcd":
            new_text, new_answer, ok = shuffle_mcq(text, answer, np.random.default_rng(seed))
            assert ok
            new = parse_mcq(new_text)
            assert new is not None
            assert new.prefix == mcq.prefix
            assert new.options[" abcd".index(new_answer) - 1] == mcq.options["abcd".index(answer)]
            assert sorted(new.options) == sorted(mcq.options)


def test_mcq_unparseable_falls_back() -> None:
    rng = np.random.default_rng(0)
    assert shuffle_mcq("Is it a) b?", "a", rng) == ("Is it a) b?", "a", False)
    dup = "Pick: a) x, b) x, c) y, d) z"
    assert shuffle_mcq(dup, "c", rng) == (dup, "c", False)  # duplicate options: ambiguous
    assert shuffle_mcq("Q a) w, b) x, c) y, d) z", "e", rng)[2] is False


# ------------------------------------------------------------------------- conversations
def test_leakage_rules_hold_over_50k_conversations() -> None:
    rng = np.random.default_rng(0)
    images = [_image(i, rng) for i in range(4000)] + [_bench_like(i, rng) for i in range(1000)]
    n = 0
    for pid, items in images:
        for visit in range(10):
            conv = build_conversation(items, pid, visit, CFG, seed=7)
            problems = check_leakage_rules(conv)
            assert not problems, (pid, visit, problems)
            groups = [t.group for t in conv.turns]
            if "presence" in groups:  # explicit re-check of the ordering rule
                assert not set(groups[: groups.index("presence")]) & CLASS_MENTIONING
            n += 1
    assert n == 50_000


def test_determinism_and_visit_variation() -> None:
    pid, items = _image(3, np.random.default_rng(1), adjacency=True, relpos=True)
    a = build_conversation(items, pid, 0, CFG, seed=1)
    b = build_conversation(items, pid, 0, CFG, seed=1)
    assert a == b
    c = build_conversation(items, pid, 0, CFG, seed=2)
    visits = [build_conversation(items, pid, v, CFG, seed=1) for v in range(3)]
    assert c != a or visits[1] != a
    # over three visits each triple group cycles through binary-yes / binary-no / MCQ
    for g in ("area", "count", "adjacency"):
        ids = {t.qa_id for v in visits for t in v.turns if t.group == g}
        assert len(ids) == 3, (g, ids)


def test_selection_statistics_match_config() -> None:
    rng = np.random.default_rng(3)
    images = [_image(i, rng) for i in range(6000)]
    kinds: Counter[str] = Counter()
    adj: Counter[str] = Counter()
    presence = caption = turns = 0
    modality: Counter[tuple[str, ...]] = Counter()
    first: Counter[str] = Counter()
    for pid, items in images:
        conv = build_conversation(items, pid, 0, CFG, seed=11)
        turns += sum(t.kind != "caption" for t in conv.turns)
        modality[conv.composites] += 1
        first[conv.turns[0].group or "caption"] += 1
        for t in conv.turns:
            if t.group == "area":
                kinds["mcq" if t.kind == "mcq" else f"binary_{t.answer}"] += 1
            if t.group == "adjacency" and t.kind == "binary":
                adj[t.answer] += 1
            presence += t.group == "presence"
            caption += t.kind == "caption"
    n = len(images)
    assert abs(kinds["mcq"] / n - 0.5) < 0.03
    assert abs(kinds["binary_yes"] / n - 0.25) < 0.03
    assert abs(presence / n - 0.7) < 0.03
    assert abs(caption / n - 0.2) < 0.02
    assert abs(adj["yes"] / (adj["yes"] + adj["no"]) - 0.44) < 0.04  # balanced-ish, not 2:1
    assert abs(modality[("rgb", "nir_swir", "sar")] / n - 0.6) < 0.03
    assert abs(modality[("sar",)] / n - 0.05) < 0.015
    assert 5.4 < turns / n < 6.1
    assert all(first[g] > 0.05 * n for g in ("area", "count", "season", "geo"))  # every
    # category is regularly the first turn (= single-turn inference format)


def test_caption_always_last_and_weighted() -> None:
    cfg = ConversationConfig(p_caption=1.0, caption_turn_weight=0.5)
    rng = np.random.default_rng(5)
    for i in range(200):
        pid, items = _image(i, rng)
        conv = build_conversation(items, pid, i % 3, cfg, seed=0)
        assert conv.turns[-1].kind == "caption"
        assert conv.turns[-1].weight == 0.5
        assert all(t.weight == 1.0 for t in conv.turns[:-1])


def test_options_shuffled_but_answers_still_correct() -> None:
    rng = np.random.default_rng(9)
    pid, items = _image(1, rng, adjacency=True, relpos=True)
    by_id = {it.id: it for it in items}
    moved = 0
    for visit in range(30):
        for t in build_conversation(items, pid, visit, CFG, seed=0).turns:
            if t.kind != "mcq":
                continue
            orig = by_id[t.qa_id]
            o, n = parse_mcq(orig.input), parse_mcq(t.question)
            assert o is not None
            assert n is not None
            assert t.shuffled
            assert n.options["abcd".index(t.answer)] == o.options["abcd".index(orig.output)]
            moved += t.answer != orig.output
    assert moved > 0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="summing to 1"):
        ConversationConfig(group_pick_probs={"binary_yes": 0.5, "binary_no": 0.5, "mcq": 0.5})
    with pytest.raises(ValueError, match="composites"):
        ConversationConfig(modality_dropout={"all": 0.5, "rgb+red_edge": 0.5})
