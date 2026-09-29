"""``analyze()``: the single entry point used by the API, tests and the eval harness."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from satquery import __version__
from satquery.agent.artifacts import write_artifacts
from satquery.agent.classifier import ClassifierContext, RuleBasedClassifier, TaskClassifier
from satquery.agent.composer import AnswerComposer, FinalAnswer
from satquery.agent.executor import Executor
from satquery.agent.planner import Planner
from satquery.agent.trace import Artifact, ExecutionTrace, InputSummary
from satquery.errors import ErrorCode, ErrorInfo, SatQueryError
from satquery.io.validate import InputBundle, InputOverrides, validate_inputs
from satquery.tools.registry import Registry, default_registry

log = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace: ExecutionTrace
    bundle: InputBundle | None = None

    @property
    def answer(self) -> FinalAnswer | None:
        return self.trace.answer

    @property
    def artifacts(self) -> list[Artifact]:
        return self.trace.artifacts

    @property
    def ok(self) -> bool:
        return self.trace.status in ("ok", "partial")


_DEFAULT_REGISTRY: Registry | None = None


def _registry() -> Registry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = default_registry()
    return _DEFAULT_REGISTRY


def analyze(
    paths: list[Path],
    query: str,
    overrides: list[InputOverrides | None] | None = None,
    *,
    benchmark: bool = False,
    out_dir: Path | None = None,
    classifier: TaskClassifier | None = None,
    registry: Registry | None = None,
    request_id: str | None = None,
) -> AnalysisResult:
    """Validate -> classify -> plan -> execute -> compose. Never raises for user errors:
    failures are recorded in the trace (``status`` rejected / error)."""
    start = time.perf_counter()
    registry = registry or _registry()
    classifier = classifier or RuleBasedClassifier()
    trace = ExecutionTrace(
        request_id=request_id or uuid.uuid4().hex,
        created_at=datetime.now(UTC),
        satquery_version=__version__,
        query=query,
        inputs=InputSummary(files=[Path(p).name for p in paths]),
        registry_digest=registry.digest(),
    )
    bundle: InputBundle | None = None
    try:
        try:
            bundle = validate_inputs(paths, overrides, benchmark=benchmark)
        except SatQueryError:
            # Still classify, so a rejected trace shows what was asked.
            trace.task = classifier.classify(query, ClassifierContext())
            raise
        trace.inputs = InputSummary.of(trace.inputs.files, bundle)
        trace.warnings.extend(bundle.warnings)

        ctx = ClassifierContext(
            config=bundle.config, modalities=[layer.modality for layer in bundle.layers.values()]
        )
        trace.task = classifier.classify(query, ctx)
        plan = Planner(registry).plan(trace.task, bundle)
        trace.plan = plan
        trace.warnings.extend(plan.warnings)

        results = Executor(registry).execute(plan, bundle, query)
        trace.record_steps(plan, results)
        for res in results:
            if res.output is not None:
                trace.warnings.extend(f"{res.step_id}: {w}" for w in res.output.warnings)
            if res.error is not None:
                trace.errors.append(res.error)

        trace.answer = AnswerComposer().compose(trace.task, plan, results, bundle)
        failed = any(r.status != "ok" for r in results)
        answered = trace.answer.confidence_basis != "required step failed"
        trace.status = "ok" if not failed else "partial" if answered else "error"
        if out_dir is not None:
            trace.artifacts = write_artifacts(out_dir, bundle, results, trace.answer)
    except SatQueryError as exc:
        trace.status = "rejected"
        trace.errors.append(exc.info)
    except Exception as exc:
        log.exception("internal error in analyze()")
        trace.status = "error"
        trace.errors.append(
            ErrorInfo(code=ErrorCode.INTERNAL_ERROR, message=f"{type(exc).__name__}: {exc}")
        )
    trace.total_runtime_ms = round((time.perf_counter() - start) * 1000, 3)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trace.json").write_text(trace.to_json(), encoding="utf-8")
    return AnalysisResult(trace=trace, bundle=bundle)
