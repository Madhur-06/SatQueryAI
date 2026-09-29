"""End-to-end ``analyze()`` on synthetic data: answers, confidence, artifacts and trace shape."""

import json
from pathlib import Path

import pytest

from satquery.agent.pipeline import analyze
from satquery.agent.trace import SCHEMA_VERSION, ExecutionTrace
from satquery.errors import ErrorCode

TRACE_KEYS = {
    "schema_version", "request_id", "created_at", "satquery_version", "status", "query",
    "inputs", "task", "plan", "steps", "answer", "artifacts", "warnings", "errors",
    "registry_digest", "total_runtime_ms",
}  # fmt: skip
STEP_KEYS = {
    "step_id", "tool", "version", "role", "purpose", "layers", "inputs", "params", "status",
    "runtime_ms", "output_summary", "confidence", "is_mock", "error",
}  # fmt: skip


def test_water_area_single_optical(tmp_path: Path, optical_4band) -> None:
    res = analyze([optical_4band.path], "How much area is covered by water?", out_dir=tmp_path)
    t = res.trace
    assert t.status == "ok"
    assert t.answer is not None
    assert t.answer.value == pytest.approx(optical_4band.truth_hectares("water"), rel=0.02)
    assert t.answer.unit == "ha"
    assert "ha" in t.answer.text
    assert 0.5 < t.answer.confidence <= 0.95
    assert "capped" in t.answer.confidence_basis
    assert not t.answer.is_mock
    kinds = {a.kind for a in t.artifacts}
    assert kinds == {"quicklook", "mask_png", "overlay", "geotiff", "geojson"}
    for artifact in t.artifacts:
        assert (tmp_path / artifact.filename).exists()
    saved = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert saved["request_id"] == t.request_id


def test_sar_water_yes_no(optical_4band, sar_db) -> None:
    t = analyze([sar_db.path], "Is there any water in this image?").trace
    assert t.inputs.config == "single_sar"
    assert t.answer is not None
    assert t.answer.value is True
    assert t.answer.text.startswith("Yes")


def test_fusion_reports_agreement(optical_4band, sar_db) -> None:
    t = analyze([optical_4band.path, sar_db.path], "How much water is there?").trace
    assert t.task is not None
    assert t.task.task == "fusion_extract"
    assert t.answer is not None
    assert t.answer.facts["agreement_iou"] > 0.85
    assert "agree" in t.answer.text
    assert {e.key for e in t.answer.evidence} == {"agree", "optical_only", "sar_only"}


def test_builtup_increase(temporal_pair) -> None:
    t1, t2 = temporal_pair
    t = analyze([t2.path, t1.path], "Has built-up area increased?").trace
    assert t.status == "ok"
    assert t.answer is not None
    assert t.answer.value is True
    assert t.answer.facts["gain"] == pytest.approx(t2.truth_hectares("builtup_gain"), rel=0.03)
    assert t.answer.facts["cross_check_iou"] > 0.9
    assert "2020-03-01" in t.answer.text
    assert "2024-03-01" in t.answer.text
    assert [s.tool for s in t.steps] == [
        "ndbi_builtup_mask", "ndbi_builtup_mask", "mask_compare", "area_stats", "change_cva",
    ]  # fmt: skip


def test_builtup_decrease_answers_no(temporal_pair) -> None:
    t = analyze([p.path for p in temporal_pair], "Did the built-up area decrease?").trace
    assert t.answer is not None
    assert t.answer.value is False
    assert t.answer.text.startswith("No")


def test_caption_is_mock(optical_4band) -> None:
    t = analyze([optical_4band.path], "Describe this image").trace
    assert t.answer is not None
    assert t.answer.is_mock
    assert t.answer.confidence == 0.0
    assert t.answer.text.startswith("[MOCK]")
    assert t.steps[0].is_mock


def test_rejected_change_query_with_one_image(tmp_path: Path, optical_4band) -> None:
    res = analyze([optical_4band.path], "Has the lake shrunk since 2020?", out_dir=tmp_path)
    t = res.trace
    assert not res.ok
    assert t.status == "rejected"
    assert t.errors[0].code is ErrorCode.TASK_INPUT_MISMATCH
    assert t.task is not None  # still shows what was asked
    assert t.plan is None
    assert t.answer is None
    assert (tmp_path / "trace.json").exists()


def test_rejected_invalid_input_still_classified(optical_4band, optical_disjoint) -> None:
    t = analyze([optical_4band.path, optical_disjoint.path], "What changed?").trace
    assert t.status == "rejected"
    assert t.errors[0].code is ErrorCode.NO_OVERLAP
    assert t.task is not None
    assert t.inputs.files == [optical_4band.path.name, optical_disjoint.path.name]


def test_trace_shape_is_stable(temporal_pair) -> None:
    t = analyze([p.path for p in temporal_pair], "Has built-up area increased?").trace
    data = json.loads(t.to_json())
    assert set(data) == TRACE_KEYS
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["created_at"].endswith("Z") or "+00:00" in data["created_at"]
    for step in data["steps"]:
        assert set(step) == STEP_KEYS
        assert step["params"] is not None  # effective params, defaults included
        json.dumps(step["output_summary"])
    layer = data["inputs"]["layers"][0]
    assert layer["band_map"]["swir1"] == 5  # 1-based for humans
    assert len(layer["sha256"]) == 64
    assert data["task"]["classifier"] == "rule_based_v0"
    assert data["plan"]["recipe"] == "class_change"
    # Round-trips through the model.
    assert ExecutionTrace.model_validate_json(t.to_json()) == t
