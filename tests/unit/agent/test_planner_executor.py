from typing import ClassVar

import pytest

from satquery.agent.classifier import TaskSpec
from satquery.agent.executor import Executor
from satquery.agent.plan import Plan, PlanStep
from satquery.agent.planner import Planner
from satquery.errors import ErrorCode, SatQueryError
from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.tools.registry import Registry, default_registry
from satquery.types import AnswerType, Capability, InputConfig, TargetClass, TaskType


def spec(task, targets=(), answer=AnswerType.YES_NO, direction=None) -> TaskSpec:
    return TaskSpec(
        task=task,
        targets=list(targets),
        answer_type=answer,
        direction=direction,
        confidence=1.0,
        classifier="test",
    )


def tools_of(plan: Plan) -> list[str]:
    return [s.tool for s in plan.steps]


# ------------------------------------------------------------------ planner


def test_plan_water_single_optical(registry, opt_bundle) -> None:
    plan = Planner(registry).plan(spec(TaskType.VQA, [TargetClass.WATER]), opt_bundle)
    assert plan.recipe == "mapping"
    assert tools_of(plan) == ["ndwi_water_mask", "area_stats"]
    assert plan.steps[1].inputs == {"source": plan.steps[0].id}
    assert plan.rationale


def test_plan_water_single_sar(registry, sar_bundle) -> None:
    plan = Planner(registry).plan(spec(TaskType.VQA, [TargetClass.WATER]), sar_bundle)
    assert tools_of(plan) == ["sar_water_otsu", "area_stats"]


def test_plan_fusion(registry, optsar_bundle) -> None:
    plan = Planner(registry).plan(spec(TaskType.FUSION_EXTRACT), optsar_bundle)
    assert plan.recipe == "fusion"
    assert tools_of(plan) == [
        "ndwi_water_mask",
        "sar_water_otsu",
        "optsar_fusion_rules",
        "area_stats",
    ]
    assert any("defaulting to water" in r for r in plan.rationale)


def test_plan_fusion_falls_back_to_single_source(registry, optsar_bundle) -> None:
    plan = Planner(registry).plan(
        spec(TaskType.FUSION_EXTRACT, [TargetClass.VEGETATION]), optsar_bundle
    )
    assert plan.recipe == "single_source"
    assert tools_of(plan) == ["ndvi_vegetation_mask", "area_stats"]
    assert any("No SAR tool" in w for w in plan.warnings)


def test_plan_builtup_increase(registry, temporal_bundle) -> None:
    """The spec's example: ndbi on both dates -> compare -> area (+ CVA cross-check)."""
    plan = Planner(registry).plan(
        spec(TaskType.CHANGE_VQA, [TargetClass.BUILTUP], direction="increase"), temporal_bundle
    )
    assert plan.recipe == "class_change"
    assert tools_of(plan) == [
        "ndbi_builtup_mask",
        "ndbi_builtup_mask",
        "mask_compare",
        "area_stats",
        "change_cva",
    ]
    assert [s.layers for s in plan.steps[:2]] == [["t1"], ["t2"]]
    assert plan.steps[2].inputs == {"before": plan.steps[0].id, "after": plan.steps[1].id}
    assert plan.steps[4].role == "cross_check"


def test_plan_change_description_adds_language_step(registry, temporal_bundle) -> None:
    plan = Planner(registry).plan(
        spec(TaskType.CHANGE_DESC, answer=AnswerType.FREE_TEXT), temporal_bundle
    )
    assert plan.recipe == "generic_change"
    assert tools_of(plan) == ["change_cva", "area_stats", "change_vqa"]
    assert plan.steps[-1].is_mock


def test_plan_sar_change_without_class_tool_falls_back(registry, sar_temporal_bundle) -> None:
    bundle, _ = sar_temporal_bundle
    plan = Planner(registry).plan(spec(TaskType.CHANGE_VQA, [TargetClass.BUILTUP]), bundle)
    assert plan.recipe == "generic_change"
    assert tools_of(plan) == ["sar_logratio_change", "area_stats"]
    assert any("generic change" in w for w in plan.warnings)


@pytest.mark.parametrize(
    ("task", "tool"),
    [
        (TaskType.VQA, "vlm_vqa"),
        (TaskType.CAPTION, "vlm_caption"),
        (TaskType.GROUNDING, "grounding"),
    ],
)
def test_plan_vlm_tasks_use_mocks(registry, opt_bundle, task, tool) -> None:
    plan = Planner(registry).plan(spec(task, answer=AnswerType.FREE_TEXT), opt_bundle)
    assert plan.recipe == "vlm"
    assert tools_of(plan) == [tool]
    assert plan.steps[0].is_mock


def test_plan_rejects_change_with_one_image(registry, opt_bundle) -> None:
    with pytest.raises(SatQueryError) as exc:
        Planner(registry).plan(spec(TaskType.CHANGE_VQA, [TargetClass.WATER]), opt_bundle)
    assert exc.value.code is ErrorCode.TASK_INPUT_MISMATCH


def test_plan_rejects_builtup_area_without_swir(registry, opt_bundle) -> None:
    area_question = spec(TaskType.VQA, [TargetClass.BUILTUP], answer=AnswerType.AREA)
    with pytest.raises(SatQueryError) as exc:
        Planner(registry).plan(area_question, opt_bundle)
    assert exc.value.code is ErrorCode.BAND_ROLE_MISSING
    problems = exc.value.info.details["candidates"]["ndbi_builtup_mask"]
    assert problems == ["optical: no swir1 band identified"]


def test_plan_reports_modality_mismatch(registry, sar_bundle) -> None:
    area_question = spec(TaskType.VQA, [TargetClass.VEGETATION], answer=AnswerType.AREA)
    with pytest.raises(SatQueryError) as exc:
        Planner(registry).plan(area_question, sar_bundle)
    assert exc.value.code is ErrorCode.MODALITY_MISMATCH


@pytest.mark.parametrize("answer", [AnswerType.YES_NO, AnswerType.COUNT, AnswerType.FREE_TEXT])
def test_plan_falls_back_to_vlm_when_target_cannot_be_mapped(registry, opt_bundle, answer) -> None:
    """RGB/4-band inputs: non-area questions about built-up go to the VLM, with a warning."""
    plan = Planner(registry).plan(spec(TaskType.VQA, [TargetClass.BUILTUP], answer), opt_bundle)
    assert plan.recipe == "vlm"
    assert tools_of(plan) == ["vlm_vqa"]
    assert any("vision-language" in w for w in plan.warnings)
    assert any("fallback" in r for r in plan.rationale)


def test_better_tool_is_picked_up_without_planner_changes(opt_bundle) -> None:
    class FancyWater(Tool):
        name = "fancy_water"
        version = "9.9"
        description = "higher-priority water model"
        tasks = frozenset({TaskType.VQA})
        input_configs = frozenset({InputConfig.SINGLE_OPT})
        produces = frozenset({Capability.MASK_WATER})
        requires = ToolRequirements(n_layers=1, modality="optical")
        priority = 50

        def run(self, inp, params):  # pragma: no cover - not executed
            raise NotImplementedError

    reg = default_registry()
    reg.register(FancyWater())
    plan = Planner(reg).plan(spec(TaskType.VQA, [TargetClass.WATER]), opt_bundle)
    assert plan.steps[0].tool == "fancy_water"
    assert "alternatives: ndwi_water_mask" in plan.rationale[0]


# ------------------------------------------------------------------ executor


class _Boom(Tool):
    name = "boom"
    version = "0"
    description = "always crashes"
    tasks = frozenset(TaskType)
    input_configs = frozenset(InputConfig)
    produces: ClassVar = frozenset({Capability.MASK_WATER})
    requires = ToolRequirements(n_layers=1)

    def run(self, inp: ToolInput, params: ToolParams) -> ToolOutput:
        raise RuntimeError("kaboom")


def _step(id_, tool, capability, role, layers=(), inputs=None, params=None) -> PlanStep:
    return PlanStep(
        id=id_, tool=tool, tool_version="x", capability=capability, role=role,
        layers=list(layers), inputs=inputs or {}, params=params or {}, purpose="test",
    )  # fmt: skip


def test_executor_isolates_failures_and_skips_dependents(opt_bundle) -> None:
    reg: Registry = default_registry()
    reg.register(_Boom())
    plan = Plan(
        recipe="test",
        steps=[
            _step("a", "boom", Capability.MASK_WATER, "mask", ["optical"]),
            _step("b", "area_stats", Capability.AREA_STATS, "measure", inputs={"source": "a"}),
            _step("c", "ndwi_water_mask", Capability.MASK_WATER, "mask", ["optical"]),
        ],
    )
    a, b, c = Executor(reg).execute(plan, opt_bundle)
    assert a.status == "failed"
    assert a.error is not None
    assert a.error.code is ErrorCode.TOOL_FAILED
    assert "kaboom" in a.error.message
    assert b.status == "skipped"
    assert c.status == "ok"
    assert c.params_effective == {
        "threshold": 0.0,
        "method": "fixed",
        "min_region_px": 4,
        "variant": "auto",
    }


def test_executor_rejects_params_outside_schema(registry, opt_bundle) -> None:
    plan = Plan(
        recipe="test",
        steps=[
            _step("a", "ndwi_water_mask", Capability.MASK_WATER, "mask", ["optical"],
                  params={"threshold": 5}),
        ],
    )  # fmt: skip
    (a,) = Executor(registry).execute(plan, opt_bundle)
    assert a.status == "failed"
    assert a.error is not None
    assert a.error.code is ErrorCode.INVALID_PARAMS
