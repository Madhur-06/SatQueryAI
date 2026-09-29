import pytest

from satquery.agent.classifier import ClassifierContext, RuleBasedClassifier
from satquery.types import AnswerType, InputConfig, TargetClass, TaskType

W, B, V = TargetClass.WATER, TargetClass.BUILTUP, TargetClass.VEGETATION
SINGLE = ClassifierContext(config=InputConfig.SINGLE_OPT)
TEMPORAL = ClassifierContext(config=InputConfig.PAIR_TEMPORAL)
OPTSAR = ClassifierContext(config=InputConfig.PAIR_OPT_SAR)

CASES = [
    # query, context, task, targets, answer type, direction
    ("Is there any water in this image?", SINGLE, TaskType.VQA, [W], AnswerType.YES_NO, None),
    ("How much area is covered by water?", SINGLE, TaskType.VQA, [W], AnswerType.AREA, None),
    ("What percentage of the scene is urban?", SINGLE, TaskType.VQA, [B], AnswerType.AREA, None),
    ("How many lakes are there?", SINGLE, TaskType.VQA, [W], AnswerType.COUNT, None),
    ("Are there buildings near the river?", SINGLE, TaskType.VQA, [W, B], AnswerType.YES_NO, None),
    ("What is the extent of forest cover?", SINGLE, TaskType.VQA, [V], AnswerType.AREA, None),
    ("Is the area flooded?", SINGLE, TaskType.VQA, [W], AnswerType.YES_NO, None),
    ("What is the dominant land cover?", SINGLE, TaskType.VQA, [], AnswerType.FREE_TEXT, None),
    ("Is this a residential area?", SINGLE, TaskType.VQA, [], AnswerType.YES_NO, None),
    ("Describe this image.", SINGLE, TaskType.CAPTION, [], AnswerType.FREE_TEXT, None),
    ("Give me a short caption", SINGLE, TaskType.CAPTION, [], AnswerType.FREE_TEXT, None),
    ("Summarise the scene", SINGLE, TaskType.CAPTION, [], AnswerType.FREE_TEXT, None),
    ("Where is the lake?", SINGLE, TaskType.GROUNDING, [W], AnswerType.FREE_TEXT, None),
    ("Locate the airport", SINGLE, TaskType.GROUNDING, [], AnswerType.FREE_TEXT, None),
    ("Highlight the settlements", SINGLE, TaskType.GROUNDING, [B], AnswerType.FREE_TEXT, None),
    ("Has built-up area increased?", TEMPORAL, TaskType.CHANGE_VQA, [B], AnswerType.YES_NO,
     "increase"),
    ("Has the lake shrunk?", TEMPORAL, TaskType.CHANGE_VQA, [W], AnswerType.YES_NO, "decrease"),
    ("How much forest was lost?", TEMPORAL, TaskType.CHANGE_VQA, [V], AnswerType.AREA,
     "decrease"),
    ("How many new buildings appeared?", TEMPORAL, TaskType.CHANGE_VQA, [B], AnswerType.COUNT,
     "increase"),
    ("Did anything change between the two dates?", TEMPORAL, TaskType.CHANGE_VQA, [],
     AnswerType.YES_NO, "any"),
    ("Describe the changes", TEMPORAL, TaskType.CHANGE_DESC, [], AnswerType.FREE_TEXT, "any"),
    ("What changed?", TEMPORAL, TaskType.CHANGE_DESC, [], AnswerType.FREE_TEXT, "any"),
    ("Is there water?", TEMPORAL, TaskType.CHANGE_VQA, [W], AnswerType.YES_NO, "any"),
    # a change question on a single image still classifies as change (planner rejects it)
    ("Has the river expanded since 2019?", SINGLE, TaskType.CHANGE_VQA, [W], AnswerType.YES_NO,
     "increase"),
    ("Map the flooded area", OPTSAR, TaskType.FUSION_EXTRACT, [W], AnswerType.AREA, None),
    ("How much water is there?", OPTSAR, TaskType.FUSION_EXTRACT, [W], AnswerType.AREA, None),
    ("Combine optical and SAR to extract water bodies", OPTSAR, TaskType.FUSION_EXTRACT, [W],
     AnswerType.FREE_TEXT, None),
    ("Fuse both sensors", OPTSAR, TaskType.FUSION_EXTRACT, [], AnswerType.FREE_TEXT, None),
    ("What does this image show?", OPTSAR, TaskType.CAPTION, [], AnswerType.FREE_TEXT, None),
    ("Is there any road?", OPTSAR, TaskType.VQA, [], AnswerType.YES_NO, None),
]  # fmt: skip


@pytest.mark.parametrize(("query", "ctx", "task", "targets", "answer", "direction"), CASES)
def test_rule_based_classifier(query, ctx, task, targets, answer, direction) -> None:
    spec = RuleBasedClassifier().classify(query, ctx)
    assert spec.task is task, spec.rationale
    assert spec.targets == targets
    assert spec.answer_type is answer
    assert spec.direction == direction
    assert spec.classifier == "rule_based_v0"
    assert spec.rationale


def test_temporal_input_without_change_words_has_lower_confidence() -> None:
    spec = RuleBasedClassifier().classify("Is there water?", TEMPORAL)
    assert spec.confidence < 0.9
    assert any("without change wording" in r for r in spec.rationale)


def test_taskspec_is_a_tool_call_schema() -> None:
    """An LLM classifier can use this schema for function calling."""
    from satquery.agent.classifier import TaskSpec

    schema = TaskSpec.model_json_schema()
    assert set(schema["required"]) >= {"task", "answer_type", "confidence", "classifier"}
