"""Executor: runs a plan step by step, isolating failures."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from satquery.agent.plan import Plan
from satquery.errors import ErrorCode, ErrorInfo, SatQueryError
from satquery.io.validate import InputBundle
from satquery.tools.base import ToolInput, ToolOutput
from satquery.tools.registry import Registry

log = logging.getLogger(__name__)


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str
    tool: str
    status: Literal["ok", "failed", "skipped"]
    params_effective: dict[str, Any] | None = None
    """Validated parameters including defaults (what actually ran)."""
    output: ToolOutput | None = None
    error: ErrorInfo | None = None
    runtime_ms: float = 0.0


class Executor:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def execute(
        self, plan: Plan, bundle: InputBundle, query: str | None = None
    ) -> list[StepResult]:
        results: dict[str, StepResult] = {}
        for step in plan.steps:
            tool = self.registry.get(step.tool)
            not_ok = [sid for sid in step.inputs.values() if results[sid].status != "ok"]
            if not_ok:
                results[step.id] = StepResult(
                    step_id=step.id,
                    tool=step.tool,
                    status="skipped",
                    error=ErrorInfo(
                        code=ErrorCode.TOOL_FAILED,
                        message=f"Skipped because upstream step(s) {not_ok} did not succeed.",
                    ),
                )
                continue
            start = time.perf_counter()
            effective = None
            try:
                effective = tool.parse_params(step.params).model_dump(mode="json")
                inp = ToolInput(
                    bundle=bundle,
                    layers=step.layers,
                    upstream={name: results[sid].output for name, sid in step.inputs.items()},  # type: ignore[misc]
                    query=query,
                    capability=step.capability,
                )
                output = tool.invoke(inp, step.params)
                results[step.id] = StepResult(
                    step_id=step.id,
                    tool=step.tool,
                    status="ok",
                    params_effective=effective,
                    output=output,
                    runtime_ms=output.runtime_ms,
                )
            except SatQueryError as exc:
                results[step.id] = self._failed(step.id, step.tool, effective, exc.info, start)
            except Exception as exc:  # a tool bug must not take down the request
                log.exception("tool %s failed", step.tool)
                info = ErrorInfo(
                    code=ErrorCode.TOOL_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    details={"tool": step.tool},
                )
                results[step.id] = self._failed(step.id, step.tool, effective, info, start)
        return [results[s.id] for s in plan.steps]

    @staticmethod
    def _failed(
        step_id: str, tool: str, effective: dict | None, info: ErrorInfo, start: float
    ) -> StepResult:
        return StepResult(
            step_id=step_id,
            tool=tool,
            status="failed",
            params_effective=effective,
            error=info,
            runtime_ms=(time.perf_counter() - start) * 1000,
        )
