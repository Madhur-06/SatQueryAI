"""Mock tools: correct interfaces and clearly-labelled placeholder outputs where models go later.

Their answers are deliberately trivial priors ("yes" for yes/no, "0" for counts), so
that running the eval harness with a mock gives a meaningful floor baseline.
"""

from __future__ import annotations

import re

from pydantic import Field

from satquery.tools.base import GeoBox, Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.types import Capability, InputConfig, TaskType

MOCK_PREFIX = "[MOCK]"
_YES_NO = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.I
)
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.I)


def prior_answer(question: str | None) -> str:
    """Majority-class style prior used by the mocks."""
    q = question or ""
    if _COUNT.search(q):
        return "0"
    if _YES_NO.search(q):
        return "yes"
    return "unknown"


class MockParams(ToolParams):
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    """Accepted for interface parity with the real VLM; ignored."""


def _scene(inp: ToolInput) -> str:
    ref = inp.bundle.reference
    res = f"{ref.meta.res[0]:g}" if ref.crs else "?"
    return f"{ref.modality} image, {ref.shape[0]}x{ref.shape[1]} px, pixel size {res}"


class MockVQA(Tool):
    name = "vlm_vqa"
    version = "0.0.1-mock"
    description = "Placeholder for a vision-language VQA model. Answers with a fixed prior."
    tasks = frozenset({TaskType.VQA})
    input_configs = frozenset(InputConfig)
    produces = frozenset({Capability.ANSWER})
    requires = ToolRequirements(n_layers=1)
    Params = MockParams
    is_mock = True

    def run(self, inp: ToolInput, params: MockParams) -> ToolOutput:
        answer = prior_answer(inp.query)
        return ToolOutput(
            text=f"{MOCK_PREFIX} No VLM is configured yet; placeholder answer: {answer}.",
            values={"answer": answer, "mock": True},
            confidence=0.0,
            confidence_basis="mock tool",
        )


class MockCaption(Tool):
    name = "vlm_caption"
    version = "0.0.1-mock"
    description = "Placeholder for a captioning model. Describes only the image metadata."
    tasks = frozenset({TaskType.CAPTION})
    input_configs = frozenset(InputConfig)
    produces = frozenset({Capability.CAPTION})
    requires = ToolRequirements(n_layers=1)
    Params = MockParams
    is_mock = True

    def run(self, inp: ToolInput, params: MockParams) -> ToolOutput:
        scene = _scene(inp)
        caption = f"{'An' if scene[0] in 'aeiou' else 'A'} {scene}."
        return ToolOutput(
            text=f"{MOCK_PREFIX} {caption} (A captioning model will describe the content.)",
            values={"answer": caption, "mock": True},
            confidence=0.0,
            confidence_basis="mock tool",
        )


class MockGrounding(Tool):
    name = "grounding"
    version = "0.0.1-mock"
    description = "Placeholder for text-guided grounding. Returns one full-image box, score 0."
    tasks = frozenset({TaskType.GROUNDING})
    input_configs = frozenset(InputConfig)
    produces = frozenset({Capability.BOXES})
    requires = ToolRequirements(n_layers=1)
    Params = MockParams
    is_mock = True

    def run(self, inp: ToolInput, params: MockParams) -> ToolOutput:
        h, w = inp.bundle.reference.shape
        label = f"{MOCK_PREFIX} {inp.query or 'object'}"
        box = GeoBox(label=label, bbox_px=(0, 0, w, h), score=0.0)
        return ToolOutput(
            text=f"{MOCK_PREFIX} No grounding model is configured; returning the full image.",
            boxes=[box],
            values={"answer": "full image", "mock": True},
            confidence=0.0,
            confidence_basis="mock tool",
        )


class MockChangeVQA(Tool):
    name = "change_vqa"
    version = "0.0.1-mock"
    description = "Placeholder for a bi-temporal change-VQA model. Answers with a fixed prior."
    tasks = frozenset({TaskType.CHANGE_VQA, TaskType.CHANGE_DESC})
    input_configs = frozenset({InputConfig.PAIR_TEMPORAL})
    produces = frozenset({Capability.CHANGE_ANSWER})
    requires = ToolRequirements(n_layers=2)
    Params = MockParams
    is_mock = True

    def run(self, inp: ToolInput, params: MockParams) -> ToolOutput:
        answer = prior_answer(inp.query)
        t1, t2 = inp.layer(0), inp.layer(1)
        dates = f"{t1.acquired or '?'} -> {t2.acquired or '?'}"
        return ToolOutput(
            text=f"{MOCK_PREFIX} No change-VQA model is configured ({dates}); "
            f"placeholder answer: {answer}.",
            values={"answer": answer, "mock": True},
            confidence=0.0,
            confidence_basis="mock tool",
        )


MOCK_TOOLS = (MockVQA, MockCaption, MockGrounding, MockChangeVQA)
