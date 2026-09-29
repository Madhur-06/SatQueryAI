"""AnswerComposer: tool outputs -> final answer, confidence and evidence.

Confidence heuristic (v0, deliberately simple and fully reported in the trace):

* start from the confidence of the step that produced the primary mask (threshold
  margin x histogram bimodality for the classical tools, see each tool);
* opt-SAR fusion: the fusion tool already scales by optical/SAR agreement (IoU);
* class change: multiply by ``0.5 + 0.5 x IoU`` between the class gain/loss mask and an
  independent change detector (CVA); without that cross-check, multiply by 0.8;
* any failed step that the answer relies on -> no answer, confidence 0;
* answers produced by a mock tool have confidence 0 and ``is_mock = True``;
* answers from classical (threshold-based) tools are capped at 0.95: a hand-set rule
  is never certain, however clean the histogram looks.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from satquery.agent.classifier import TaskSpec
from satquery.agent.executor import StepResult
from satquery.agent.plan import Plan, PlanStep, StepRole
from satquery.io.validate import InputBundle
from satquery.tools.base import ToolOutput
from satquery.types import AnswerType

CLASSICAL_MAX_CONFIDENCE = 0.95
PRESENCE_MIN_PX = 10
PRESENCE_MIN_FRACTION = 0.001
LABELS = {"water": "water", "builtup": "built-up area", "vegetation": "vegetation"}


class Evidence(BaseModel):
    kind: Literal["mask", "boxes"]
    label: str
    step_id: str
    key: str = "mask"
    """Mask key inside the step output."""
    layer: str
    """Bundle layer the evidence should be drawn on."""


class FinalAnswer(BaseModel):
    text: str
    answer_type: AnswerType
    value: bool | int | float | str | None = None
    unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_basis: str
    facts: dict[str, Any] = Field(default_factory=dict)
    """Key numbers behind the answer (for UI tables and evaluation)."""
    evidence: list[Evidence] = Field(default_factory=list)
    is_mock: bool = False


def _fmt_area(ha: float | None, px: int) -> str:
    return f"{ha:,.2f} ha" if ha is not None else f"{px:,} px"


def _pct(fraction: float) -> str:
    return f"{100 * fraction:.1f}%"


class _Ctx:
    def __init__(self, plan: Plan, results: list[StepResult], bundle: InputBundle) -> None:
        self.plan = plan
        self.results = {r.step_id: r for r in results}
        self.bundle = bundle
        self.n_valid = int(bundle.reference.valid.sum())

    def ok(self, role: StepRole) -> list[tuple[PlanStep, ToolOutput]]:
        out = []
        for step in self.plan.by_role(role):
            res = self.results.get(step.id)
            if res is not None and res.status == "ok" and res.output is not None:
                out.append((step, res.output))
        return out

    def first(self, role: StepRole) -> tuple[PlanStep, ToolOutput] | None:
        found = self.ok(role)
        return found[0] if found else None

    def significant(self, pixels: int) -> bool:
        return pixels >= max(PRESENCE_MIN_PX, PRESENCE_MIN_FRACTION * self.n_valid)

    def failures(self) -> list[str]:
        return [
            f"{r.step_id}: {r.error.message}"
            for r in self.results.values()
            if r.status != "ok" and r.error is not None
        ]

    def dates(self) -> str:
        t1, t2 = self.bundle.layers.get("t1"), self.bundle.layers.get("t2")
        if t1 and t2 and t1.acquired and t2.acquired:
            return f"between {t1.acquired} and {t2.acquired}"
        return "between the two dates"


class AnswerComposer:
    def compose(
        self, spec: TaskSpec, plan: Plan, results: list[StepResult], bundle: InputBundle
    ) -> FinalAnswer:
        ctx = _Ctx(plan, results, bundle)
        compose = {
            "mapping": self._mapping,
            "single_source": self._mapping,
            "fusion": self._fusion,
            "class_change": self._class_change,
            "generic_change": self._generic_change,
            "vlm": self._vlm,
        }[plan.recipe]
        answer = compose(spec, ctx)
        if answer is None:
            failures = ctx.failures()
            reason = failures[0] if failures else "no tool produced a result"
            return FinalAnswer(
                text=f"The analysis could not be completed ({reason}).",
                answer_type=spec.answer_type,
                confidence=0.0,
                confidence_basis="required step failed",
                facts={"failures": failures},
            )
        answer.confidence = float(np.clip(answer.confidence, 0.0, 1.0))
        if plan.recipe != "vlm" and answer.confidence > CLASSICAL_MAX_CONFIDENCE:
            answer.confidence = CLASSICAL_MAX_CONFIDENCE
            cap = CLASSICAL_MAX_CONFIDENCE
            answer.confidence_basis += f" (capped at {cap} for rule-based tools)"
        lang = ctx.first("language")
        if lang is not None and plan.recipe != "vlm":
            step, out = lang
            answer.facts["language_model"] = {"tool": step.tool, "is_mock": out.is_mock}
            if not out.is_mock and out.text:
                answer.text += f" Model description: {out.text}"
        return answer

    # ------------------------------------------------------------------ recipes

    def _presence_text(self, spec: TaskSpec, label: str, present: bool, area: str, pct: str,
                       regions: int) -> tuple[str, bool | int | str]:  # fmt: skip
        if spec.answer_type is AnswerType.YES_NO:
            if present:
                return f"Yes: {label} covers {area} ({pct} of the scene) in {regions} region(s).", (
                    True
                )
            return f"No: no significant {label} was detected ({area}, {pct}).", False
        if spec.answer_type is AnswerType.COUNT:
            return f"{regions} separate {label} region(s) found, {area} in total ({pct}).", regions
        text = f"{label.capitalize()} covers {area} ({pct} of the scene) in {regions} region(s)."
        return text, area

    def _mapping(self, spec: TaskSpec, ctx: _Ctx) -> FinalAnswer | None:
        mask = ctx.first("mask")
        measure = ctx.first("measure")
        if mask is None or measure is None:
            return None
        mask_step, mask_out = mask
        v = measure[1].values
        label = LABELS.get(str(v.get("class")), str(v.get("class")))
        px, ha, frac = int(v["mask_pixels"]), v["mask_hectares"], float(v["mask_fraction"])
        regions = int(v["mask_regions"])
        area = _fmt_area(ha, px)
        text, value = self._presence_text(
            spec, label, ctx.significant(px), area, _pct(frac), regions
        )
        if spec.answer_type in (AnswerType.AREA, AnswerType.FREE_TEXT):
            value = ha if ha is not None else px
        if spec.task.value == "grounding":
            text = f"Found {regions} {label} region(s) ({area}); they are outlined on the overlay."
            value = regions
        return FinalAnswer(
            text=text,
            answer_type=spec.answer_type,
            value=value,
            unit=None if spec.answer_type in (AnswerType.YES_NO, AnswerType.COUNT)
            else ("ha" if ha is not None else "px"),
            confidence=mask_out.confidence or 0.0,
            confidence_basis=f"{mask_step.tool}: {mask_out.confidence_basis}",
            facts={"class": v.get("class"), "hectares": ha, "pixels": px, "fraction": frac,
                   "regions": regions, "source_layer": mask_step.layers[0]},
            evidence=[Evidence(kind="mask", label=label, step_id=mask_step.id,
                               layer=mask_step.layers[0])],
        )  # fmt: skip

    def _fusion(self, spec: TaskSpec, ctx: _Ctx) -> FinalAnswer | None:
        fuse, measure = ctx.first("fuse"), ctx.first("measure")
        if fuse is None or measure is None:
            return None
        fuse_step, fuse_out = fuse
        v, fv = measure[1].values, fuse_out.values
        label = LABELS.get(str(fv.get("class")), str(fv.get("class")))
        px, ha, frac = int(v["mask_pixels"]), v["mask_hectares"], float(v["mask_fraction"])
        text, value = self._presence_text(
            spec, label, ctx.significant(px), _fmt_area(ha, px), _pct(frac), int(v["mask_regions"])
        )
        agreement = float(fv["agreement_iou"])
        text += (
            f" Optical and SAR agree on {_fmt_area(v['agree_hectares'], int(v['agree_pixels']))} "
            f"({agreement:.0%} of all detected pixels); optical only: "
            f"{_fmt_area(v['optical_only_hectares'], int(v['optical_only_pixels']))}, SAR only: "
            f"{_fmt_area(v['sar_only_hectares'], int(v['sar_only_pixels']))}."
        )
        if spec.answer_type in (AnswerType.AREA, AnswerType.FREE_TEXT):
            value = ha if ha is not None else px
        evidence = [
            Evidence(kind="mask", label=f"{label} (both sensors)", step_id=fuse_step.id,
                     key="agree", layer="optical"),
            Evidence(kind="mask", label=f"{label} (optical only)", step_id=fuse_step.id,
                     key="optical_only", layer="optical"),
            Evidence(kind="mask", label=f"{label} (SAR only)", step_id=fuse_step.id,
                     key="sar_only", layer="sar"),
        ]  # fmt: skip
        return FinalAnswer(
            text=text,
            answer_type=spec.answer_type,
            value=value,
            unit="ha" if spec.answer_type is AnswerType.AREA and ha is not None else None,
            confidence=fuse_out.confidence or 0.0,
            confidence_basis=f"{fuse_step.tool}: {fuse_out.confidence_basis}",
            facts={"class": fv.get("class"), "hectares": ha, "pixels": px, "fraction": frac,
                   "agreement_iou": agreement, "rule": fv.get("rule")},
            evidence=evidence,
        )  # fmt: skip

    def _class_change(self, spec: TaskSpec, ctx: _Ctx) -> FinalAnswer | None:
        cmp, measure = ctx.first("compare"), ctx.first("measure")
        if cmp is None or measure is None:
            return None
        cmp_step, cmp_out = cmp
        v = measure[1].values
        label = LABELS.get(str(cmp_out.values.get("class")), str(cmp_out.values.get("class")))
        gain_px, loss_px = int(v["gain_pixels"]), int(v["loss_pixels"])
        net_px = int(cmp_out.values["net_change_pixels"])
        gain_ha, loss_ha = v["gain_hectares"], v["loss_hectares"]
        before_ha, after_ha = v["before_hectares"], v["after_hectares"]
        georef = gain_ha is not None
        net = (after_ha - before_ha) if georef else net_px
        unit = "ha" if georef else "px"

        increased = ctx.significant(net_px)
        decreased = ctx.significant(-net_px)
        changed = ctx.significant(gain_px + loss_px)
        trend = "increased" if increased else "decreased" if decreased else "stayed about the same"
        detail = (
            f"{label.capitalize()} {trend} {ctx.dates()}: from "
            f"{_fmt_area(before_ha, int(v['before_pixels']))} to "
            f"{_fmt_area(after_ha, int(v['after_pixels']))} "
            f"(gained {_fmt_area(gain_ha, gain_px)}, lost {_fmt_area(loss_ha, loss_px)}, "
            f"{int(v['gain_regions'])} new region(s))."
        )
        value: bool | int | float
        if spec.answer_type is AnswerType.YES_NO:
            by_direction = {"increase": increased, "decrease": decreased}
            yes = by_direction.get(spec.direction or "any", changed)
            value = yes
            text = f"{'Yes' if yes else 'No'}. {detail}"
        elif spec.answer_type is AnswerType.COUNT:
            value = int(v["gain_regions"])
            text = (
                f"{value} new {label} region(s) appeared (connected regions, not individual "
                f"objects). {detail}"
            )
        else:
            value = round(float(net), 4)
            text = detail

        confidence = cmp_out.confidence or 0.0
        basis = f"{cmp_step.tool}: {cmp_out.confidence_basis}"
        facts: dict[str, Any] = {
            "class": cmp_out.values.get("class"), "unit": unit, "net_change": net,
            "gain": gain_ha if georef else gain_px, "loss": loss_ha if georef else loss_px,
            "before": before_ha, "after": after_ha,
        }  # fmt: skip
        cross = ctx.first("cross_check")
        evidence = [
            Evidence(kind="mask", label=f"{label} gained", step_id=cmp_step.id, key="gain",
                     layer="t2"),
            Evidence(kind="mask", label=f"{label} lost", step_id=cmp_step.id, key="loss",
                     layer="t1"),
        ]  # fmt: skip
        if cross is not None:
            cross_step, cross_out = cross
            class_change = cmp_out.masks["mask"].array.astype(bool)
            spectral = cross_out.masks["mask"].array.astype(bool)
            union = (class_change | spectral).sum()
            agreement = float((class_change & spectral).sum() / union) if union else 1.0
            confidence *= 0.5 + 0.5 * agreement
            basis += f"; x (0.5 + 0.5 x IoU {agreement:.2f} with {cross_step.tool})"
            facts["cross_check_iou"] = round(agreement, 4)
            text += (
                f" An independent spectral change detector ({cross_step.tool}) agrees "
                f"on {agreement:.0%} of the changed area."
            )
            evidence.append(
                Evidence(kind="mask", label="any spectral change", step_id=cross_step.id,
                         layer="t2")
            )  # fmt: skip
        else:
            confidence *= 0.8
            basis += "; x 0.8 (no independent cross-check)"
        return FinalAnswer(
            text=text,
            answer_type=spec.answer_type,
            value=value,
            unit=unit if spec.answer_type in (AnswerType.AREA, AnswerType.FREE_TEXT) else None,
            confidence=confidence,
            confidence_basis=basis,
            facts=facts,
            evidence=evidence,
        )

    def _generic_change(self, spec: TaskSpec, ctx: _Ctx) -> FinalAnswer | None:
        change, measure = ctx.first("change"), ctx.first("measure")
        if change is None or measure is None:
            return None
        step, out = change
        v = measure[1].values
        px, ha, frac = int(v["mask_pixels"]), v["mask_hectares"], float(v["mask_fraction"])
        changed = ctx.significant(px)
        text = f"About {_fmt_area(ha, px)} ({_pct(frac)} of the scene) changed {ctx.dates()}"
        if "increase_pixels" in v:
            up = _fmt_area(v["increase_hectares"], int(v["increase_pixels"]))
            down = _fmt_area(v["decrease_hectares"], int(v["decrease_pixels"]))
            text += f"; backscatter increased on {up} and decreased on {down}"
        text += "."
        value: bool | float | int = ha if ha is not None else px
        if spec.answer_type is AnswerType.YES_NO:
            value = changed
            text = f"{'Yes' if changed else 'No'}. {text}"
        elif spec.answer_type is AnswerType.COUNT:
            value = int(v["mask_regions"])
            text = f"{value} changed region(s). {text}"
        return FinalAnswer(
            text=text,
            answer_type=spec.answer_type,
            value=value,
            unit="ha" if isinstance(value, float) and ha is not None else None,
            confidence=out.confidence or 0.0,
            confidence_basis=f"{step.tool}: {out.confidence_basis}",
            facts={"hectares": ha, "pixels": px, "fraction": frac, "regions": v["mask_regions"]},
            evidence=[Evidence(kind="mask", label="change", step_id=step.id, layer="t2")],
        )

    def _vlm(self, spec: TaskSpec, ctx: _Ctx) -> FinalAnswer | None:
        lang = ctx.first("language")
        if lang is None:
            return None
        step, out = lang
        evidence = []
        if out.boxes:
            evidence.append(Evidence(kind="boxes", label="grounding", step_id=step.id,
                                     layer=step.layers[0]))  # fmt: skip
        if out.is_mock:
            confidence, basis = 0.0, "mock tool (placeholder output)"
        elif out.confidence is not None:
            confidence, basis = out.confidence, out.confidence_basis or step.tool
        else:
            confidence, basis = 0.5, f"{step.tool}: uncalibrated zero-shot model (fixed 0.5)"
        return FinalAnswer(
            text=out.text or "",
            answer_type=spec.answer_type,
            value=out.values.get("answer"),
            confidence=confidence,
            confidence_basis=basis,
            facts={"tool": step.tool},
            evidence=evidence,
            is_mock=out.is_mock,
        )
