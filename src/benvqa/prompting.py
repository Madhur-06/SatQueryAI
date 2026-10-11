"""Prompt constants shared by training, evaluation and analysis.

Changing any of these changes the token sequences the adapter is trained on, so the
inference side (``eval_bench``, Phase 3 tool) must import them from here.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an Earth-observation analyst. Images of one 1.2 km scene are labelled by "
    "sensor. Answer yes/no questions with yes or no, and multiple-choice questions with "
    "the option letter only."
)

IMAGE_TAGS: dict[str, str] = {
    "rgb": "[S2 RGB]",
    "nir_swir": "[S2 NIR-SWIR B8/B11/B12]",
    "sar": "[S1 SAR VV/VH]",
    "red_edge": "[S2 Red-edge B5/B6/B7]",
}

TOKENIZER_ID = "Qwen/Qwen3.5-4B"
"""Tokenizer/processor source for every model variant (``unsloth/`` ships a different one)."""
