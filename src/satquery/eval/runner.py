"""Run one tool (or the whole pipeline) over a benchmark subset and score it.

Every report records the tool name/version, whether it is a mock, the registry digest,
the git commit and the exact subset (size + seed), so later fine-tuned models can be
compared like for like against the zero-shot / mock baselines.
"""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from satquery import __version__
from satquery.errors import SatQueryError
from satquery.eval.datasets.base import BenchmarkDataset, BenchmarkSample
from satquery.eval.metrics import caption_scores, grounding_scores, vqa_accuracy
from satquery.io.validate import InputOverrides, validate_inputs
from satquery.tools.base import Tool, ToolInput
from satquery.tools.registry import Registry, default_registry
from satquery.types import Capability, TaskType

PIPELINE = "pipeline"
TASK_CAPABILITY = {
    TaskType.VQA: Capability.ANSWER,
    TaskType.CAPTION: Capability.CAPTION,
    TaskType.GROUNDING: Capability.BOXES,
    TaskType.CHANGE_VQA: Capability.CHANGE_ANSWER,
    TaskType.CHANGE_DESC: Capability.CHANGE_ANSWER,
}


class SampleRecord(BaseModel):
    id: str
    task: TaskType
    question: str | None
    question_type: str | None
    references: list[str] = Field(default_factory=list)
    prediction: str | None = None
    bbox_gt: tuple[float, float, float, float] | None = None
    bbox_pred: tuple[float, float, float, float] | None = None
    error: str | None = None
    runtime_ms: float = 0.0


class EvalReport(BaseModel):
    benchmark: str
    split: str
    tool: str
    tool_version: str
    is_mock: bool
    mode: Literal["tool", "pipeline"]
    subset: int | None
    seed: int
    n_samples: int
    n_errors: int
    metrics: dict[str, Any]
    created_at: datetime
    satquery_version: str
    registry_digest: str
    git_commit: str | None
    runtime_s: float
    records: list[SampleRecord]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _overrides(dataset: BenchmarkDataset, n: int) -> list[InputOverrides | None] | None:
    modality = getattr(dataset, "modality", None)
    return [InputOverrides(modality=modality)] * n if modality else None


def predict_with_tool(
    tool: Tool, sample: BenchmarkSample, dataset: BenchmarkDataset
) -> SampleRecord:
    record = SampleRecord(
        id=sample.id, task=sample.task, question=sample.question,
        question_type=sample.question_type, references=sample.answers, bbox_gt=sample.bbox,
    )  # fmt: skip
    bundle = validate_inputs(sample.images, _overrides(dataset, len(sample.images)), benchmark=True)
    inp = ToolInput(
        bundle=bundle,
        layers=list(bundle.layers),
        query=sample.question,
        capability=TASK_CAPABILITY.get(sample.task),
    )
    out = tool.invoke(inp, {})
    record.prediction = _as_text(out.values.get("answer", out.text))
    if sample.task is TaskType.GROUNDING and out.boxes:
        h, w = bundle.reference.shape
        x1, y1, x2, y2 = out.boxes[0].bbox_px
        record.bbox_pred = (100 * x1 / w, 100 * y1 / h, 100 * x2 / w, 100 * y2 / h)
    return record


def predict_with_pipeline(sample: BenchmarkSample, dataset: BenchmarkDataset) -> SampleRecord:
    from satquery.agent.pipeline import analyze

    record = SampleRecord(
        id=sample.id, task=sample.task, question=sample.question,
        question_type=sample.question_type, references=sample.answers, bbox_gt=sample.bbox,
    )  # fmt: skip
    trace = analyze(
        sample.images,
        sample.question or "Describe this image",
        _overrides(dataset, len(sample.images)),
        benchmark=True,
    ).trace
    if trace.answer is None:
        record.error = trace.errors[0].message if trace.errors else "no answer"
        return record
    value = trace.answer.value
    if sample.task is TaskType.CAPTION or value is None:
        value = trace.answer.text
    record.prediction = _as_text(value)
    return record


def score(records: list[SampleRecord], task: TaskType) -> dict[str, Any]:
    if task in (TaskType.VQA, TaskType.CHANGE_VQA):
        return vqa_accuracy([(r.prediction, r.references, r.question_type) for r in records])
    if task in (TaskType.CAPTION, TaskType.CHANGE_DESC):
        preds = {r.id: r.prediction or "" for r in records}
        refs = {r.id: r.references for r in records if r.references}
        return caption_scores(preds, refs)
    if task is TaskType.GROUNDING:
        return grounding_scores([(r.bbox_pred, r.bbox_gt) for r in records if r.bbox_gt])
    raise ValueError(f"no metric for task {task}")


def run_eval(
    dataset: BenchmarkDataset,
    tool_name: str,
    *,
    subset: int | None = None,
    seed: int = 0,
    registry: Registry | None = None,
    progress: bool = False,
) -> EvalReport:
    registry = registry or default_registry()
    start = time.perf_counter()
    samples = dataset.subset(subset, seed)
    tasks = {s.task for s in samples}
    if len(tasks) > 1:
        raise ValueError(f"mixed tasks in one run: {tasks}")
    task = tasks.pop() if tasks else next(iter(dataset.tasks))

    if tool_name == PIPELINE:
        tool, version, is_mock, mode = None, __version__, False, "pipeline"
    else:
        tool = registry.get(tool_name)
        needed = TASK_CAPABILITY[task]
        if needed not in tool.produces:
            raise ValueError(
                f"tool '{tool_name}' does not support {task}: it produces "
                f"{sorted(tool.produces)}, the benchmark needs {needed}"
            )
        version, is_mock, mode = tool.version, tool.is_mock, "tool"

    records: list[SampleRecord] = []
    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        try:
            if tool is None:
                record = predict_with_pipeline(sample, dataset)
            else:
                record = predict_with_tool(tool, sample, dataset)
        except (SatQueryError, ValueError, OSError) as exc:
            record = SampleRecord(
                id=sample.id, task=sample.task, question=sample.question,
                question_type=sample.question_type, references=sample.answers,
                bbox_gt=sample.bbox, error=f"{type(exc).__name__}: {exc}",
            )  # fmt: skip
        record.runtime_ms = (time.perf_counter() - t0) * 1000
        records.append(record)
        if progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)}")

    return EvalReport(
        benchmark=dataset.name,
        split=dataset.split,
        tool=tool_name,
        tool_version=version,
        is_mock=is_mock,
        mode=mode,  # type: ignore[arg-type]
        subset=subset,
        seed=seed,
        n_samples=len(records),
        n_errors=sum(r.error is not None for r in records),
        metrics=score(records, task) if records else {},
        created_at=datetime.now(UTC),
        satquery_version=__version__,
        registry_digest=registry.digest(),
        git_commit=_git_commit(),
        runtime_s=round(time.perf_counter() - start, 3),
        records=records,
    )
