"""The single ``Tool`` interface every classical, mock and learned tool implements.

A tool declares, as class attributes, what it is (name, version), what it serves
(tasks, input configurations, produced capabilities), what it needs (layer modality,
band roles, upstream outputs) and which parameters may be set (a pydantic model with
``extra="forbid"`` and bounded fields). The controller never calls models directly;
it calls ``Tool.invoke`` with a ``ToolInput`` and a plain params dict.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.georaster import GeoRaster
from satquery.io.validate import InputBundle, RasterLayer
from satquery.types import BandRole, Capability, InputConfig, TaskType


class ToolParams(BaseModel):
    """Base for tool parameters: unknown keys are rejected, values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NoParams(ToolParams):
    pass


class GeoBox(BaseModel):
    label: str
    bbox_px: tuple[float, float, float, float]
    """(x_min, y_min, x_max, y_max) in pixels of the reference grid."""
    score: float | None = None
    geometry: dict[str, Any] | None = None
    """WGS84 GeoJSON polygon, when the grid is georeferenced."""


class ToolOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    masks: dict[str, GeoRaster] = Field(default_factory=dict)
    """Boolean masks; ``"mask"`` is the primary one."""
    rasters: dict[str, GeoRaster] = Field(default_factory=dict)
    """Continuous outputs (index values, change magnitude...)."""
    geojson: dict[str, dict[str, Any]] = Field(default_factory=dict)
    boxes: list[GeoBox] = Field(default_factory=list)
    values: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    confidence: float | None = None
    confidence_basis: str | None = None
    runtime_ms: float = 0.0
    is_mock: bool = False
    warnings: list[str] = Field(default_factory=list)

    @property
    def mask(self) -> GeoRaster | None:
        return self.masks.get("mask")

    def summary(self) -> dict[str, Any]:
        """JSON-safe digest for the trace (no pixel data)."""
        out: dict[str, Any] = {}
        if self.text is not None:
            out["text"] = self.text
        if self.values:
            out["values"] = {
                k: round(v, 4) if isinstance(v, float) else v for k, v in self.values.items()
            }
        if self.masks:
            out["masks"] = {
                k: {**m.summary(), "true_fraction": round(float(np.mean(m.array)), 4)}
                for k, m in self.masks.items()
            }
        if self.rasters:
            out["rasters"] = {k: r.summary() for k, r in self.rasters.items()}
        if self.geojson:
            out["geojson"] = {k: len(v.get("features", [])) for k, v in self.geojson.items()}
        if self.boxes:
            out["boxes"] = [b.model_dump(exclude={"geometry"}) for b in self.boxes]
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
            out["confidence_basis"] = self.confidence_basis
        if self.warnings:
            out["warnings"] = self.warnings
        out["is_mock"] = self.is_mock
        return out


class ToolInput(BaseModel):
    """What a tool receives: the bundle, which layers to use, and upstream outputs by name."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundle: InputBundle
    layers: list[str]
    upstream: dict[str, ToolOutput] = Field(default_factory=dict)
    query: str | None = None

    def layer(self, i: int = 0) -> RasterLayer:
        return self.bundle.layer(self.layers[i])

    def upstream_mask(self, name: str, key: str = "mask") -> GeoRaster:
        out = self.upstream.get(name)
        if out is None or key not in out.masks:
            raise SatQueryError(
                ErrorCode.TOOL_FAILED,
                f"Missing upstream mask '{name}.{key}'.",
                details={"available": {k: list(v.masks) for k, v in self.upstream.items()}},
            )
        return out.masks[key]


class ToolRequirements(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_layers: int = 0
    """How many bundle layers the tool reads (0 for tools that only consume upstream outputs)."""
    modality: Literal["optical", "sar", "any"] = "any"
    """Modality family every input layer must have."""
    bands: tuple[tuple[BandRole, ...], ...] = ()
    """Required band roles; each inner tuple lists acceptable alternatives."""
    upstream: dict[str, Capability | None] = Field(default_factory=dict)
    """Named upstream inputs and the capability each must provide (None = any mask)."""


class ToolSpec(BaseModel):
    """Public, JSON-exportable description of a tool (registry export, trace, report)."""

    name: str
    version: str
    description: str
    tasks: list[TaskType]
    input_configs: list[InputConfig]
    produces: list[Capability]
    requires: ToolRequirements
    params_schema: dict[str, Any]
    is_mock: bool
    available: bool
    unavailable_reason: str | None


class Tool(ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]
    tasks: ClassVar[frozenset[TaskType]]
    input_configs: ClassVar[frozenset[InputConfig]]
    produces: ClassVar[frozenset[Capability]]
    requires: ClassVar[ToolRequirements] = ToolRequirements()
    Params: ClassVar[type[ToolParams]] = NoParams
    is_mock: ClassVar[bool] = False
    priority: ClassVar[int] = 0
    """Tie-breaker when several tools provide the same capability (higher wins)."""

    def is_available(self) -> tuple[bool, str | None]:
        return True, None

    def requirement_problems(self, layers: list[RasterLayer]) -> list[tuple[ErrorCode, str]]:
        """Why this tool cannot run on ``layers``, as ``(code, message)`` (empty = OK)."""
        req = self.requires
        problems: list[tuple[ErrorCode, str]] = []
        if len(layers) < req.n_layers:
            msg = f"needs {req.n_layers} layer(s), got {len(layers)}"
            problems.append((ErrorCode.TASK_INPUT_MISMATCH, msg))
        for layer in layers[: req.n_layers]:
            if req.modality != "any" and layer.modality.family != req.modality:
                msg = f"{layer.key}: needs {req.modality} imagery, got {layer.modality}"
                problems.append((ErrorCode.MODALITY_MISMATCH, msg))
                continue  # band roles are meaningless for the wrong modality
            for alternatives in req.bands:
                if not any(r in layer.band_map for r in alternatives):
                    msg = f"{layer.key}: no {' or '.join(alternatives)} band identified"
                    problems.append((ErrorCode.BAND_ROLE_MISSING, msg))
        return problems

    def missing_requirements(self, layers: list[RasterLayer]) -> list[str]:
        return [msg for _, msg in self.requirement_problems(layers)]

    @abstractmethod
    def run(self, inp: ToolInput, params: Any) -> ToolOutput: ...

    def parse_params(self, raw: dict[str, Any] | None) -> ToolParams:
        try:
            return self.Params.model_validate(raw or {})
        except ValidationError as exc:
            raise SatQueryError(
                ErrorCode.INVALID_PARAMS,
                f"Invalid parameters for tool '{self.name}'.",
                details={
                    "tool": self.name,
                    "errors": [
                        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                        for e in exc.errors()
                    ],
                },
            ) from exc

    def invoke(self, inp: ToolInput, raw_params: dict[str, Any] | None = None) -> ToolOutput:
        """Validate params and requirements, run, and stamp runtime / mock flag."""
        ok, reason = self.is_available()
        if not ok:
            raise SatQueryError(
                ErrorCode.TOOL_UNAVAILABLE,
                f"Tool '{self.name}' is not available: {reason}",
                details={"tool": self.name, "reason": reason},
            )
        params = self.parse_params(raw_params)
        layers = [inp.bundle.layer(k) for k in inp.layers]
        if problems := self.requirement_problems(layers):
            code = problems[0][0]
            raise SatQueryError(
                code,
                f"Tool '{self.name}' cannot run on these inputs.",
                details={"tool": self.name, "problems": [msg for _, msg in problems]},
                hint=(
                    "Provide a band_map override." if code is ErrorCode.BAND_ROLE_MISSING else None
                ),
            )
        for name in self.requires.upstream:
            if name not in inp.upstream:
                raise SatQueryError(
                    ErrorCode.TOOL_FAILED,
                    f"Tool '{self.name}' is missing upstream input '{name}'.",
                    details={"tool": self.name, "expected": list(self.requires.upstream)},
                )
        start = time.perf_counter()
        out = self.run(inp, params)
        out.runtime_ms = (time.perf_counter() - start) * 1000
        out.is_mock = out.is_mock or self.is_mock
        return out

    @classmethod
    def spec(cls, available: bool = True, reason: str | None = None) -> ToolSpec:
        return ToolSpec(
            name=cls.name,
            version=cls.version,
            description=cls.description,
            tasks=sorted(cls.tasks),
            input_configs=sorted(cls.input_configs),
            produces=sorted(cls.produces),
            requires=cls.requires,
            params_schema=cls.Params.model_json_schema(),
            is_mock=cls.is_mock,
            available=available,
            unavailable_reason=reason,
        )
