"""Plan data model and the registry-driven step builder used by recipes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.validate import InputBundle
from satquery.tools.registry import Registry
from satquery.types import Capability

StepRole = Literal["mask", "fuse", "compare", "change", "measure", "cross_check", "language"]


class PlanStep(BaseModel):
    id: str
    tool: str
    tool_version: str
    capability: Capability
    role: StepRole
    layers: list[str] = Field(default_factory=list)
    inputs: dict[str, str] = Field(default_factory=dict)
    """Upstream input name -> id of the step that produces it."""
    params: dict[str, Any] = Field(default_factory=dict)
    purpose: str
    is_mock: bool = False


class Plan(BaseModel):
    recipe: str
    steps: list[PlanStep]
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def step(self, step_id: str) -> PlanStep:
        return next(s for s in self.steps if s.id == step_id)

    def by_role(self, role: StepRole) -> list[PlanStep]:
        return [s for s in self.steps if s.role == role]


class PlanBuilder:
    """Accumulates steps; each ``add`` asks the registry for the best tool for a capability."""

    def __init__(self, registry: Registry, bundle: InputBundle) -> None:
        self.registry = registry
        self.bundle = bundle
        self.steps: list[PlanStep] = []
        self.rationale: list[str] = []
        self.warnings: list[str] = []

    def can(self, capability: Capability, layers: list[str] | tuple[str, ...] = ()) -> bool:
        return bool(self._candidates(capability, layers))

    def _candidates(self, capability: Capability, layers) -> list:
        raster_layers = [self.bundle.layer(k) for k in layers]
        return self.registry.find(
            capability=capability, config=self.bundle.config, layers=raster_layers
        )

    def add(
        self,
        capability: Capability,
        *,
        role: StepRole,
        purpose: str,
        layers: list[str] | tuple[str, ...] = (),
        inputs: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        candidates = self._candidates(capability, layers)
        if not candidates:
            raise self._no_tool_error(capability, list(layers), purpose)
        tool = candidates[0]
        step_id = f"s{len(self.steps) + 1}_{tool.name}"
        self.steps.append(
            PlanStep(
                id=step_id,
                tool=tool.name,
                tool_version=tool.version,
                capability=capability,
                role=role,
                layers=list(layers),
                inputs=inputs or {},
                params=params or {},
                purpose=purpose,
                is_mock=tool.is_mock,
            )
        )
        others = [t.name for t in candidates[1:]]
        note = f" (alternatives: {', '.join(others)})" if others else ""
        self.rationale.append(f"{step_id}: {purpose} -> {tool.name} v{tool.version}{note}")
        return step_id

    def _no_tool_error(self, capability: Capability, layers: list[str], purpose: str):
        raster_layers = [self.bundle.layer(k) for k in layers]
        producers = self.registry.find(capability=capability, include_unavailable=True)
        problems: dict[str, list[str]] = {}
        codes: set[ErrorCode] = set()
        for tool in producers:
            available, reason = tool.is_available()
            if not available:
                problems[tool.name] = [f"unavailable: {reason}"]
                codes.add(ErrorCode.TOOL_UNAVAILABLE)
                continue
            found = tool.requirement_problems(raster_layers)
            problems[tool.name] = [msg for _, msg in found]
            codes.update(code for code, _ in found)
            if self.bundle.config not in tool.input_configs:
                problems[tool.name].append(f"does not accept {self.bundle.config} inputs")
        details = {"capability": capability, "layers": layers, "candidates": problems}
        if ErrorCode.BAND_ROLE_MISSING in codes:
            return SatQueryError(
                ErrorCode.BAND_ROLE_MISSING,
                f"Cannot {purpose}: the required spectral bands are not in the input.",
                details=details,
                hint="Built-up mapping needs a SWIR band; water needs green + NIR (or SWIR). "
                "If the bands exist but are unlabelled, pass a band_map override.",
            )
        if ErrorCode.MODALITY_MISMATCH in codes and producers:
            return SatQueryError(
                ErrorCode.MODALITY_MISMATCH,
                f"Cannot {purpose} from this kind of imagery.",
                details=details,
            )
        only_unavailable = codes == {ErrorCode.TOOL_UNAVAILABLE}
        code = ErrorCode.TOOL_UNAVAILABLE if only_unavailable else ErrorCode.NO_TOOL
        return SatQueryError(code, f"No registered tool can {purpose}.", details=details)

    def build(self, recipe: str) -> Plan:
        return Plan(
            recipe=recipe, steps=self.steps, rationale=self.rationale, warnings=self.warnings
        )
