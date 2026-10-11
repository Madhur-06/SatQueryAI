import json
import sys

import pytest

from satquery.errors import ErrorCode, SatQueryError
from satquery.testing.helpers import run_tool
from satquery.tools.classical import NDWIWaterMask
from satquery.tools.registry import Registry
from satquery.types import Capability, TaskType

EXPECTED_TOOLS = {
    "area_stats",
    "change_cva",
    "change_vqa",
    "grounding",
    "mask_compare",
    "ndbi_builtup_mask",
    "ndvi_vegetation_mask",
    "ndwi_water_mask",
    "optsar_fusion_rules",
    "qwen_vl_zeroshot",
    "sar_logratio_change",
    "sar_water_otsu",
    "vlm_caption",
    "vlm_vqa",
}


def test_default_registry_contents(registry) -> None:
    assert set(registry.names) == EXPECTED_TOOLS


def test_register_duplicate_and_unknown() -> None:
    reg = Registry()
    reg.register(NDWIWaterMask())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(NDWIWaterMask())
    with pytest.raises(SatQueryError) as exc:
        reg.get("nope")
    assert exc.value.code is ErrorCode.NO_TOOL


def test_find_by_capability_respects_layer_modality(registry, opt_bundle, sar_bundle) -> None:
    opt = registry.find(capability=Capability.MASK_WATER, layers=[opt_bundle.reference])
    sar = registry.find(capability=Capability.MASK_WATER, layers=[sar_bundle.reference])
    assert [t.name for t in opt] == ["ndwi_water_mask"]
    assert [t.name for t in sar] == ["sar_water_otsu"]
    # 4-band optical has no SWIR: nothing can produce a built-up mask.
    assert registry.find(capability=Capability.MASK_BUILTUP, layers=[opt_bundle.reference]) == []


def test_find_by_task_prefers_real_then_available(registry) -> None:
    vqa = registry.find(task=TaskType.VQA, capability=Capability.ANSWER)
    assert [t.name for t in vqa] == ["vlm_vqa"]  # Qwen is disabled by default
    with_unavailable = registry.find(
        task=TaskType.VQA, capability=Capability.ANSWER, include_unavailable=True
    )
    assert [t.name for t in with_unavailable] == ["vlm_vqa", "qwen_vl_zeroshot"]


def test_export_is_json_and_digest_is_stable(registry) -> None:
    specs = registry.export_json()
    json.dumps(specs)  # serialisable
    ndwi = next(s for s in specs if s["name"] == "ndwi_water_mask")
    assert ndwi["params_schema"]["properties"]["threshold"]["maximum"] == 1.0
    assert ndwi["produces"] == ["mask:water"]
    qwen = next(s for s in specs if s["name"] == "qwen_vl_zeroshot")
    assert qwen["available"] is False
    assert "SATQUERY_VLM_ENABLED" in qwen["unavailable_reason"]
    assert all(s["is_mock"] for s in specs if s["name"] in {"vlm_vqa", "grounding"})
    assert registry.digest() == registry.digest()
    assert len(registry.digest()) == 16


@pytest.mark.parametrize(
    "params",
    [{"unknown": 1}, {"threshold": 2.0}, {"method": "magic"}, {"min_region_px": -1}],
)
def test_params_outside_schema_rejected(registry, opt_bundle, params) -> None:
    with pytest.raises(SatQueryError) as exc:
        run_tool(registry.get("ndwi_water_mask"), opt_bundle, ["optical"], params=params)
    assert exc.value.code is ErrorCode.INVALID_PARAMS
    assert exc.value.info.details["errors"]


def test_sar_params_validation(registry, sar_bundle) -> None:
    for bad in ({"window": 4}, {"clip_db": [5, -35]}):
        with pytest.raises(SatQueryError) as exc:
            run_tool(registry.get("sar_water_otsu"), sar_bundle, ["sar"], params=bad)
        assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_requirement_errors(registry, opt_bundle) -> None:
    with pytest.raises(SatQueryError) as exc:
        run_tool(registry.get("ndbi_builtup_mask"), opt_bundle, ["optical"])
    assert exc.value.code is ErrorCode.BAND_ROLE_MISSING
    with pytest.raises(SatQueryError) as exc:
        run_tool(registry.get("sar_water_otsu"), opt_bundle, ["optical"])
    assert exc.value.code is ErrorCode.MODALITY_MISMATCH
    assert exc.value.info.details["problems"] == ["optical: needs sar imagery, got optical"]
    with pytest.raises(SatQueryError, match="upstream"):
        run_tool(registry.get("area_stats"), opt_bundle)


def test_qwen_is_lazy_and_unavailable(registry, opt_bundle, monkeypatch) -> None:
    # Other tests may have imported transformers already; hide it so the check below only
    # sees an import made by the tool itself.
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    with pytest.raises(SatQueryError) as exc:
        run_tool(registry.get("qwen_vl_zeroshot"), opt_bundle, ["optical"], query="Any water?")
    assert exc.value.code is ErrorCode.TOOL_UNAVAILABLE
    assert "transformers" not in sys.modules


def test_output_summary_is_json_safe(registry, opt_bundle) -> None:
    out = run_tool(registry.get("ndwi_water_mask"), opt_bundle, ["optical"])
    summary = out.summary()
    json.dumps(summary)
    assert summary["masks"]["mask"]["crs"] == "EPSG:32643"
    assert 0 < summary["masks"]["mask"]["true_fraction"] < 1
    assert out.runtime_ms > 0
