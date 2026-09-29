"""API response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from satquery.agent.composer import FinalAnswer
from satquery.agent.trace import ExecutionTrace
from satquery.errors import ErrorInfo


class ArtifactOut(BaseModel):
    kind: str
    label: str
    url: str
    media_type: str
    layer: str | None = None
    step_id: str | None = None


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str
    answer: FinalAnswer | None
    confidence: float | None
    error: ErrorInfo | None
    artifacts: list[ArtifactOut]
    trace_url: str
    trace: ExecutionTrace


class ErrorResponse(BaseModel):
    error: ErrorInfo
    trace: ExecutionTrace | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    tools: int
    vlm: dict[str, Any]
