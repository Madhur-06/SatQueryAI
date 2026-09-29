"""Write an EvalReport as three files sharing one stem:

* ``<stem>.json``: metadata + metrics (small, meant to be committed);
* ``<stem>.records.jsonl.gz``: one JSON line per sample (predictions, references, errors);
* ``<stem>.md``: human-readable summary.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

from satquery.eval.runner import EvalReport, SampleRecord

N_EXAMPLES = 10


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "-" if value is None else str(value)


def _cell(text: str | None, limit: int = 80) -> str:
    text = (text or "").replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_markdown(report: EvalReport) -> str:
    lines = [f"# {report.benchmark} · {report.tool}", ""]
    if report.is_mock:
        lines += [
            "> **Mock tool**: these numbers are a trivial-prior floor, not a model result.",
            "",
        ]
    subset = "all" if report.subset is None else str(report.subset)
    meta = [
        ("Benchmark / split", f"{report.benchmark} / {report.split}"),
        ("Tool", f"{report.tool} v{report.tool_version} ({report.mode})"),
        ("Samples (subset, seed)", f"{report.n_samples} ({subset}, seed {report.seed})"),
        ("Errors", str(report.n_errors)),
        ("SatQuery / commit", f"{report.satquery_version} / {report.git_commit or '?'}"),
        ("Registry digest", report.registry_digest),
        ("Created", report.created_at.isoformat(timespec="seconds")),
        ("Runtime", f"{report.runtime_s:.1f} s"),
    ]
    lines += ["| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in meta], ""]

    lines += ["## Metrics", "", "| Metric | Value |", "|---|---|"]
    for key, value in report.metrics.items():
        if not isinstance(value, dict):
            lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    per_type = report.metrics.get("per_type")
    if per_type:
        lines += ["### Per question type", "", "| Type | Accuracy | n |", "|---|---|---|"]
        lines += [f"| {t} | {_fmt(v['accuracy'])} | {v['n']} |" for t, v in per_type.items()]
        lines.append("")

    lines += ["## Examples", "", "| Question | Prediction | Reference |", "|---|---|---|"]
    for r in report.records[:N_EXAMPLES]:
        ref = r.references[0] if r.references else (str(r.bbox_gt) if r.bbox_gt else "")
        pred = r.prediction if r.error is None else f"ERROR: {r.error}"
        lines.append(f"| {_cell(r.question)} | {_cell(pred)} | {_cell(ref)} |")
    return "\n".join(lines) + "\n"


def write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.strftime("%Y%m%dT%H%M%SZ")
    stem = out_dir / f"{report.benchmark}_{report.tool}_{stamp}"
    json_path = stem.with_suffix(".json")
    md_path = stem.with_suffix(".md")
    records_path = stem.with_name(stem.name + ".records.jsonl.gz")
    summary = report.model_dump_json(indent=2, exclude={"records"}) + "\n"
    json_path.write_text(summary, encoding="utf-8", newline="\n")
    with gzip.open(records_path, "wt", encoding="utf-8") as fh:
        for record in report.records:
            fh.write(record.model_dump_json() + "\n")
    md_path.write_text(to_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


def read_records(json_path: Path) -> list[SampleRecord]:
    """Per-sample records written next to a report's summary JSON."""
    records_path = json_path.with_name(json_path.stem + ".records.jsonl.gz")
    with gzip.open(records_path, "rt", encoding="utf-8") as fh:
        return [SampleRecord.model_validate_json(line) for line in fh if line.strip()]
