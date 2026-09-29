"""Query -> TaskSpec.

``TaskClassifier`` is the seam: the v0 ``RuleBasedClassifier`` uses keyword tables, and
an LLM function-calling classifier can replace it later by returning the same
``TaskSpec`` (it is a plain pydantic model, so it doubles as a tool-call JSON schema).
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from satquery.types import AnswerType, InputConfig, Modality, TargetClass, TaskType

Direction = Literal["increase", "decrease", "any"]


class TaskSpec(BaseModel):
    task: TaskType
    targets: list[TargetClass] = Field(default_factory=list)
    answer_type: AnswerType
    direction: Direction | None = None
    """For change tasks: which direction the question asks about."""
    confidence: float = Field(ge=0.0, le=1.0)
    classifier: str
    rationale: list[str] = Field(default_factory=list)


class ClassifierContext(BaseModel):
    """What a classifier may know besides the text: the validated input configuration."""

    config: InputConfig | None = None
    modalities: list[Modality] = Field(default_factory=list)


class TaskClassifier(Protocol):
    name: str

    def classify(self, query: str, ctx: ClassifierContext) -> TaskSpec: ...


def _rx(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(words) + r")", re.IGNORECASE)


TARGET_PATTERNS: dict[TargetClass, re.Pattern[str]] = {
    TargetClass.WATER: _rx(
        "water", "river", "lake", "flood", "pond", "reservoir", "inundat", "wetland",
        "sea\\b", "ocean", "canal", "stream",
    ),
    TargetClass.BUILTUP: _rx(
        "built[- ]?up", "building", "urban", "settlement", "construct", "city", "cities",
        "town", "impervious", "infrastructure", "houses",
    ),
    TargetClass.VEGETATION: _rx(
        "vegetat", "forest", "tree", "green cover", "crop", "farmland", "agricultur", "grass",
        "canopy", "plantation", "deforest",
    ),
}  # fmt: skip

CHANGE = _rx(
    "chang", "increas", "decreas", "grow", "grew", "expan", "shrank", "shrink", "shrunk", "reduc",
    "new\\b", "appear", "disappear", "lost\\b", "loss", "gain", "compared", "before",
    "after", "over time", "since", "between the (two )?(dates|images)", "differen",
    "deforest", "encroach",
)  # fmt: skip
INCREASE = _rx("increas", "grow", "grew", "expan", "more\\b", "gain", "new\\b", "appear")
DECREASE = _rx("decreas", "shrank", "shrink", "shrunk", "reduc", "less\\b", "fewer",
               "lost\\b", "loss", "disappear", "declin", "deforest")  # fmt: skip
CAPTION = _rx("describe", "caption", "summar", "overview", "what does (this|the) image show",
              "what is (in|shown in) (this|the) image")  # fmt: skip
GROUNDING = _rx("where\\b", "locate", "locali", "find\\b", "point out", "highlight", "mark\\b",
                "bounding box", "show me")  # fmt: skip
FUSION = _rx("optical and sar", "sar and optical", "fuse", "fusion", "combin", "both sensors",
             "jointly", "joint\\b")  # fmt: skip
YES_NO = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.I
)
COUNT = _rx("how many", "number of", "count")
AREA = _rx("how much", "area", "extent", "hectare", "km2", "km²", "percent", "%", "fraction",
           "proportion", "share", "cover", "size")  # fmt: skip


class RuleBasedClassifier:
    """v0 keyword classifier. Deterministic, explainable (every decision is in ``rationale``)."""

    name = "rule_based_v0"

    def classify(self, query: str, ctx: ClassifierContext) -> TaskSpec:
        q = query.strip()
        why: list[str] = []
        targets = [t for t, rx in TARGET_PATTERNS.items() if rx.search(q)]
        if targets:
            why.append(f"targets: {', '.join(targets)}")

        if YES_NO.search(q):
            answer_type = AnswerType.YES_NO
        elif COUNT.search(q):
            answer_type = AnswerType.COUNT
        elif AREA.search(q) and targets:
            answer_type = AnswerType.AREA
        else:
            answer_type = AnswerType.FREE_TEXT
        why.append(f"answer type: {answer_type}")

        is_change = bool(CHANGE.search(q))
        is_caption = bool(CAPTION.search(q))
        is_grounding = bool(GROUNDING.search(q))
        is_fusion = bool(FUSION.search(q))
        temporal_input = ctx.config is InputConfig.PAIR_TEMPORAL
        optsar_input = ctx.config is InputConfig.PAIR_OPT_SAR

        direction: Direction | None = None
        confidence = 0.9
        if is_change or (temporal_input and not is_caption):
            if is_change:
                why.append("change wording")
            else:
                why.append("bi-temporal input without change wording: treated as change")
                confidence = 0.7
            if is_caption or (answer_type is AnswerType.FREE_TEXT and not targets):
                task = TaskType.CHANGE_DESC
            else:
                task = TaskType.CHANGE_VQA
            inc, dec = bool(INCREASE.search(q)), bool(DECREASE.search(q))
            direction = "increase" if inc and not dec else "decrease" if dec and not inc else "any"
            why.append(f"direction: {direction}")
        elif is_caption and not targets:
            task = TaskType.CAPTION
            why.append("caption wording")
        elif is_grounding:
            task = TaskType.GROUNDING
            why.append("localisation wording")
        elif is_fusion or (optsar_input and targets):
            task = TaskType.FUSION_EXTRACT
            why.append("fusion wording" if is_fusion else "optical+SAR input, mappable target")
        else:
            task = TaskType.VQA
            confidence = 0.8 if targets else 0.6
            why.append("default: visual question")
        if is_caption and targets and task is TaskType.VQA:
            why.append("caption wording ignored because a target class was named")

        return TaskSpec(
            task=task,
            targets=targets,
            answer_type=answer_type,
            direction=direction,
            confidence=confidence,
            classifier=self.name,
            rationale=why,
        )
