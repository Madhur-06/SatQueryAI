"""ExecutionTrace: the auditable record of one request (what judges and reviewers see).

Stability rules: every field is typed; enums serialise as lower-case strings; numbers
are plain JSON numbers; no pixel data. Bump ``SCHEMA_VERSION`` on breaking changes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from satquery.agent.classifier import TaskSpec
from satquery.agent.composer import FinalAnswer
from satquery.agent.executor import StepResult
from satquery.agent.plan import Plan, StepRole
from satquery.errors import ErrorInfo
from satquery.io.validate import AlignmentReport, InputBundle, RasterLayer
from satquery.types import InputConfig, Modality

SCHEMA_VERSION = "1.0"


class LayerSummary(BaseModel):
    key: str
    filename: str
    sha256: str
    driver: str
    modality: Modality
    modality_source: Literal["detected", "override"]
    modality_confidence: float
    modality_reasons: list[str]
    bands: int
    dtype: str
    band_map: dict[str, int]
    """Band role -> 1-based band index."""
    band_sources: dict[str, str]
    units: str
    sar_input_scale: str | None
    crs: str | None
    resolution: tuple[float, float]
    native_shape: tuple[int, int]
    bounds: tuple[float, float, float, float]
    acquired: date | None
    acquired_source: str | None
    resampled: bool
    valid_fraction: float

    @classmethod
    def of(cls, layer: RasterLayer) -> LayerSummary:
        return cls.model_validate(layer.summary())


class GridSummary(BaseModel):
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    shape: tuple[int, int]


class InputSummary(BaseModel):
    files: list[str]
    config: InputConfig | None = None
    layers: list[LayerSummary] = Field(default_factory=list)
    alignment: AlignmentReport | None = None
    grid: GridSummary | None = None
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, files: list[str], bundle: InputBundle) -> InputSummary:
        ref = bundle.reference
        return cls(
            files=files,
            config=bundle.config,
            layers=[LayerSummary.of(layer) for layer in bundle.layers.values()],
            alignment=bundle.alignment,
            grid=GridSummary(crs=ref.crs, transform=ref.transform, shape=ref.shape),
            warnings=bundle.warnings,
        )


class StepRecord(BaseModel):
    step_id: str
    tool: str
    version: str
    role: StepRole
    purpose: str
    layers: list[str]
    inputs: dict[str, str]
    params: dict[str, Any] | None
    """Effective parameters (validated, defaults filled in)."""
    status: Literal["ok", "failed", "skipped"]
    runtime_ms: float
    output_summary: dict[str, Any] | None
    confidence: float | None
    is_mock: bool
    error: ErrorInfo | None


class Artifact(BaseModel):
    kind: Literal["quicklook", "overlay", "mask_png", "geotiff", "geojson", "trace"]
    label: str
    filename: str
    media_type: str
    layer: str | None = None
    step_id: str | None = None


class ExecutionTrace(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    created_at: datetime
    satquery_version: str
    status: Literal["ok", "partial", "rejected", "error"] = "ok"
    """ok: all steps succeeded; partial: some failed but an answer exists; rejected:
    inputs/task invalid (structured error); error: unexpected internal failure."""
    query: str
    inputs: InputSummary
    task: TaskSpec | None = None
    plan: Plan | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    answer: FinalAnswer | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorInfo] = Field(default_factory=list)
    registry_digest: str
    total_runtime_ms: float = 0.0

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def record_steps(self, plan: Plan, results: list[StepResult]) -> None:
        for step, res in zip(plan.steps, results, strict=True):
            out = res.output
            self.steps.append(
                StepRecord(
                    step_id=step.id,
                    tool=step.tool,
                    version=step.tool_version,
                    role=step.role,
                    purpose=step.purpose,
                    layers=step.layers,
                    inputs=step.inputs,
                    params=res.params_effective,
                    status=res.status,
                    runtime_ms=round(res.runtime_ms, 3),
                    output_summary=out.summary() if out else None,
                    confidence=None if out is None or out.confidence is None
                    else round(out.confidence, 4),
                    is_mock=step.is_mock,
                    error=res.error,
                )
            )  # fmt: skip
