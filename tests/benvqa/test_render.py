"""Rendering vs the official Qwen3.5 template, label masks, single-token answers, patchify.

Uses the Qwen3.5-0.8B tokenizer/processor (same family as 4B) from the local HF cache;
skipped when it is not cached (no downloads in tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from benvqa import testing as synth
from benvqa.data.conversation import ConversationConfig, QAItem, build_conversation, single_turn
from benvqa.data.render import ASSISTANT_PREFIX, IGNORE, Renderer
from benvqa.data.vision import patchify_u8, to_pixel_values, tokens_per_image

MODEL = "Qwen/Qwen3.5-0.8B"


def _transformers():
    # imported lazily: a Phase 1 test asserts transformers is never imported at collection
    return pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return _transformers().AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache")


@pytest.fixture(scope="module")
def processor():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    try:
        return _transformers().AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    except (OSError, ValueError, ImportError):
        pytest.skip(f"{MODEL} processor not available locally")


def _items(seed: int, **kw) -> tuple[str, list[QAItem]]:
    rng = np.random.default_rng(seed)
    pid, s1 = synth.patch_names(1, seed)
    rows = synth.synth_rows(pid, s1, "train", rng, **kw)

    def cat(r: dict) -> str:
        return "captioning" if r["type"] == "captioning" else r["category"]

    return pid, [
        QAItem(r["ID"], r["type"], cat(r), r["input"], r["output"])
        for r in rows
        if r["type"] != "bounding box"
    ]


def test_single_turn_matches_official_template(processor) -> None:
    from PIL import Image

    r = Renderer(processor.tokenizer, image_size=256)
    _pid, items = _items(0, adjacency=True, relpos=True)
    img = Image.fromarray(np.zeros((256, 256, 3), np.uint8))
    for item in items[:6]:
        for comps in [("rgb",), ("rgb", "nir_swir", "sar"), ("sar",)]:
            content = []
            for i, c in enumerate(comps):
                content += [
                    {"type": "text", "text": ("" if i == 0 else "\n") + _tag(c)},
                    {"type": "image", "image": img},
                ]
            content.append({"type": "text", "text": "\n" + item.input})
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": _sys()}]},
                {"role": "user", "content": content},
            ]
            out = processor.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                images_kwargs={"min_pixels": 256 * 256, "max_pixels": 256 * 256},
            )
            official = list(np.asarray(out["input_ids"]).reshape(-1))
            assert r.prompt_ids(item.input, comps) == official
            rendered = r.render(single_turn(item, comps))
            answer = [*r.encode(item.output), r.im_end]
            assert list(rendered.input_ids) == official + answer
            assert list(rendered.labels) == [IGNORE] * len(official) + answer


def _tag(c: str) -> str:
    from benvqa.prompting import IMAGE_TAGS

    return IMAGE_TAGS[c]


def _sys() -> str:
    from benvqa.prompting import SYSTEM_PROMPT

    return SYSTEM_PROMPT


def test_segmentwise_equals_whole_string_tokenization(tokenizer) -> None:
    r = Renderer(tokenizer, image_size=256)
    cfg = ConversationConfig(p_caption=0.5)
    for seed in range(300):
        pid, items = _items(seed, adjacency=seed % 3 > 0, relpos=seed % 4 == 0)
        conv = build_conversation(items, pid, seed % 3, cfg, seed=seed)
        out = r.render(conv)
        text = tokenizer.decode(out.input_ids.tolist(), skip_special_tokens=False)
        assert r.encode(text) == out.input_ids.tolist()
        # every assistant turn opens with the exact single-turn generation prefix
        assert text.count(ASSISTANT_PREFIX) == len(conv.turns)


def test_labels_and_turns(tokenizer) -> None:
    r = Renderer(tokenizer, image_size=256)
    pid, items = _items(4, adjacency=True)
    conv = build_conversation(
        items, pid, 0, ConversationConfig(p_caption=1.0, caption_turn_weight=0.25), seed=0
    )
    out = r.render(conv)
    mask = out.labels != IGNORE
    assert np.array_equal(out.labels[mask], out.input_ids[mask])
    assert np.array_equal(mask, out.label_turn >= 0)
    expected = [len(r.encode(t.answer)) + 1 for t in conv.turns]
    assert np.bincount(out.label_turn[mask], minlength=len(conv.turns)).tolist() == expected
    # each turn's labels end with <|im_end|>
    for t_idx in range(len(conv.turns)):
        last = np.flatnonzero(out.label_turn == t_idx)[-1]
        assert out.input_ids[last] == r.im_end
    assert out.turn_weight.tolist() == [1.0] * (len(conv.turns) - 1) + [0.25]
    n_img = len(conv.composites)
    assert (out.input_ids == r.image_pad).sum() == n_img * tokens_per_image(256)


def test_answers_are_single_tokens_after_prefix(tokenizer) -> None:
    r = Renderer(tokenizer, image_size=256)
    prompt = r.prompt_ids("Is there water?", ("rgb",))
    for word in ["yes", "no", "a", "b", "c", "d"]:
        full = r.encode(tokenizer.decode(prompt, skip_special_tokens=False) + word)
        assert full[: len(prompt)] == prompt, word
        assert len(full) == len(prompt) + 1, word


def test_truncation_drops_caption_first(tokenizer) -> None:
    r = Renderer(tokenizer, image_size=256, max_len=400)
    pid, items = _items(5)
    conv = build_conversation(items, pid, 0, ConversationConfig(p_caption=1.0), seed=0)
    out = r.render(conv)
    assert len(out) <= 400
    assert out.n_dropped_turns >= 1
    assert "caption" not in out.turn_kind


def test_4b_and_08b_tokenizers_agree(tokenizer) -> None:
    try:
        big = _transformers().AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-4B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen3.5-4B tokenizer not cached")
    _pid, items = _items(6, adjacency=True, relpos=True)
    texts = [it.input for it in items] + [it.output for it in items] + [ASSISTANT_PREFIX]
    for t in texts:
        assert big(t)["input_ids"] == tokenizer(t)["input_ids"]


def test_patchify_matches_official_processor(processor) -> None:
    import torch

    rng = np.random.default_rng(0)
    for size in (256, 192):
        img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        official = processor.image_processor(
            images=[img], min_pixels=size * size, max_pixels=size * size, return_tensors="pt"
        )
        tiles, grid = patchify_u8(img)
        assert list(official["image_grid_thw"][0]) == list(grid)
        ours = to_pixel_values(torch.from_numpy(tiles))
        assert ours.shape == official["pixel_values"].shape
        torch.testing.assert_close(ours, official["pixel_values"].float(), atol=1e-6, rtol=0)
        assert np.allclose(to_pixel_values(tiles), ours.numpy(), atol=1e-6)
        assert tokens_per_image(size) == grid[1] * grid[2] // 4
