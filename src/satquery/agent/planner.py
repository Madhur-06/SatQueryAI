"""Planner: (TaskSpec, InputBundle) -> ordered tool chain.

Recipes describe *what* is needed as capability chains (e.g. "water mask on t1 and t2
-> class comparison -> area"); the ``PlanBuilder`` asks the registry *which* tool
provides each capability for the concrete layers (modality, bands, availability). A
learned tool registered later with the same capability and higher priority is picked
up without touching the planner.
"""

from __future__ import annotations

from collections.abc import Callable

from satquery.agent.classifier import TaskSpec
from satquery.agent.plan import Plan, PlanBuilder
from satquery.io.validate import InputBundle, check_task_compatibility
from satquery.tools.registry import Registry
from satquery.types import TARGET_MASK, AnswerType, Capability, InputConfig, TargetClass, TaskType

Recipe = Callable[[PlanBuilder, TaskSpec, InputBundle], str]
"""Adds steps to the builder and returns the recipe name actually used."""


def _primary_layer(bundle: InputBundle) -> str:
    return "optical" if "optical" in bundle.layers else next(iter(bundle.layers))


def _first_target(b: PlanBuilder, spec: TaskSpec, default: TargetClass | None = None):
    if not spec.targets:
        if default is None:
            return None
        b.rationale.append(f"no target class named; defaulting to {default}")
        return default
    if len(spec.targets) > 1:
        b.warnings.append(
            f"Several targets named ({', '.join(spec.targets)}); v0 analyses only "
            f"'{spec.targets[0]}'."
        )
    return spec.targets[0]


def _language_step(b: PlanBuilder, spec: TaskSpec, bundle: InputBundle) -> str:
    capability = {
        TaskType.CAPTION: Capability.CAPTION,
        TaskType.GROUNDING: Capability.BOXES,
        TaskType.CHANGE_VQA: Capability.CHANGE_ANSWER,
        TaskType.CHANGE_DESC: Capability.CHANGE_ANSWER,
    }.get(spec.task, Capability.ANSWER)
    layers = ["t1", "t2"] if capability is Capability.CHANGE_ANSWER else [_primary_layer(bundle)]
    purpose = f"answer in natural language ({spec.task})"
    return b.add(capability, role="language", layers=layers, purpose=purpose)


def recipe_vlm(b: PlanBuilder, spec: TaskSpec, bundle: InputBundle) -> str:
    _language_step(b, spec, bundle)
    return "vlm"


def recipe_mapping(b: PlanBuilder, spec: TaskSpec, bundle: InputBundle) -> str:
    target = _first_target(b, spec)
    assert target is not None
    layer = _primary_layer(bundle)
    mask = b.add(
        TARGET_MASK[target], role="mask", layers=[layer], purpose=f"map {target} on {layer}"
    )
    b.add(
        Capability.AREA_STATS, role="measure", inputs={"source": mask}, purpose=f"measure {target}"
    )
    if spec.answer_type is AnswerType.FREE_TEXT:
        _language_step(b, spec, bundle)
    return "mapping"


def recipe_fusion(b: PlanBuilder, spec: TaskSpec, bundle: InputBundle) -> str:
    target = _first_target(b, spec, default=TargetClass.WATER)
    capability = TARGET_MASK[target]
    usable = [k for k in ("optical", "sar") if b.can(capability, [k])]
    if len(usable) == 2:
        opt = b.add(capability, role="mask", layers=["optical"], purpose=f"map {target} (optical)")
        sar = b.add(capability, role="mask", layers=["sar"], purpose=f"map {target} (SAR)")
        fused = b.add(
            Capability.MASK_FUSED,
            role="fuse",
            inputs={"optical": opt, "sar": sar},
            purpose=f"fuse optical and SAR {target} masks",
        )
        b.add(Capability.AREA_STATS, role="measure", inputs={"source": fused}, purpose="measure")
        return "fusion"
    if len(usable) == 1:
        only = usable[0]
        missing = "SAR" if only == "optical" else "optical"
        b.warnings.append(f"No {missing} tool can map {target}; using the {only} image only.")
        mask = b.add(capability, role="mask", layers=[only], purpose=f"map {target} ({only})")
        b.add(Capability.AREA_STATS, role="measure", inputs={"source": mask}, purpose="measure")
        return "single_source"
    b.add(capability, role="mask", layers=["optical"], purpose=f"map {target}")  # raises
    return "unreachable"


def recipe_change(b: PlanBuilder, spec: TaskSpec, bundle: InputBundle) -> str:
    target = _first_target(b, spec)
    recipe = "generic_change"
    if target is not None:
        capability = TARGET_MASK[target]
        if b.can(capability, ["t1"]) and b.can(capability, ["t2"]):
            before = b.add(
                capability, role="mask", layers=["t1"], purpose=f"map {target} at date 1"
            )
            after = b.add(capability, role="mask", layers=["t2"], purpose=f"map {target} at date 2")
            cmp = b.add(
                Capability.MASK_CLASS_CHANGE,
                role="compare",
                inputs={"before": before, "after": after},
                purpose=f"compare {target} between dates (gain / loss)",
            )
            b.add(Capability.AREA_STATS, role="measure", inputs={"source": cmp}, purpose="measure")
            if b.can(Capability.MASK_CHANGE, ["t1", "t2"]):
                b.add(
                    Capability.MASK_CHANGE,
                    role="cross_check",
                    layers=["t1", "t2"],
                    purpose="independent change detection to cross-check the class change",
                )
            recipe = "class_change"
        else:
            b.warnings.append(
                f"No tool can map {target} on these images; reporting generic change instead."
            )
    if recipe == "generic_change":
        change = b.add(
            Capability.MASK_CHANGE, role="change", layers=["t1", "t2"], purpose="detect change"
        )
        b.add(Capability.AREA_STATS, role="measure", inputs={"source": change}, purpose="measure")
    if spec.task is TaskType.CHANGE_DESC or spec.answer_type is AnswerType.FREE_TEXT:
        _language_step(b, spec, bundle)
    return recipe


class Planner:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def select_recipe(self, spec: TaskSpec, bundle: InputBundle) -> Recipe:
        if spec.task in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESC):
            return recipe_change
        if spec.task is TaskType.FUSION_EXTRACT:
            return recipe_fusion
        if spec.task in (TaskType.VQA, TaskType.GROUNDING) and spec.targets:
            if bundle.config is InputConfig.PAIR_OPT_SAR:
                return recipe_fusion
            return recipe_mapping
        return recipe_vlm

    def plan(self, spec: TaskSpec, bundle: InputBundle) -> Plan:
        """Raises ``SatQueryError`` (TASK_INPUT_MISMATCH, BAND_ROLE_MISSING, NO_TOOL...)."""
        check_task_compatibility(spec.task, bundle.config)
        builder = PlanBuilder(self.registry, bundle)
        recipe = self.select_recipe(spec, bundle)
        name = recipe(builder, spec, bundle)
        return builder.build(name)
