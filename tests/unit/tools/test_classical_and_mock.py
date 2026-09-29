from pathlib import Path

import numpy as np
import pytest
from rasterio.warp import Resampling, reproject

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.validate import validate_inputs
from satquery.testing.helpers import iou, run_tool
from satquery.testing.synthetic import write_array, write_reprojected
from satquery.tools.base import ToolOutput
from satquery.tools.mock import MOCK_PREFIX, prior_answer

# ------------------------------------------------------------------ optical indices


def test_ndwi_matches_truth(registry, opt_bundle, optical_4band) -> None:
    out = run_tool(registry.get("ndwi_water_mask"), opt_bundle, ["optical"])
    assert out.values["index"] == "ndwi"
    assert iou(out.mask.array, optical_4band.truth["water"]) > 0.95
    assert out.confidence > 0.8
    assert out.mask.crs == "EPSG:32643"
    assert out.mask.transform == opt_bundle.reference.transform
    assert "ndwi" in out.rasters


def test_ndwi_prefers_mndwi_with_swir(registry, temporal_bundle, temporal_pair) -> None:
    out = run_tool(registry.get("ndwi_water_mask"), temporal_bundle, ["t1"])
    assert out.values["index"] == "mndwi"
    assert iou(out.mask.array, temporal_pair[0].truth["water"]) > 0.95
    forced = run_tool(
        registry.get("ndwi_water_mask"), temporal_bundle, ["t1"], params={"variant": "ndwi"}
    )
    assert forced.values["index"] == "ndwi"


def test_otsu_threshold_variant(registry, opt_bundle, optical_4band) -> None:
    out = run_tool(
        registry.get("ndwi_water_mask"), opt_bundle, ["optical"], params={"method": "otsu"}
    )
    assert iou(out.mask.array, optical_4band.truth["water"]) > 0.95


def test_ndbi_builtup(registry, temporal_bundle, temporal_pair) -> None:
    before = run_tool(registry.get("ndbi_builtup_mask"), temporal_bundle, ["t1"])
    after = run_tool(registry.get("ndbi_builtup_mask"), temporal_bundle, ["t2"])
    assert before.mask.array.sum() == 0
    assert iou(after.mask.array, temporal_pair[1].truth["builtup"]) > 0.95


def test_ndvi_vegetation(registry, temporal_bundle, temporal_pair) -> None:
    out = run_tool(registry.get("ndvi_vegetation_mask"), temporal_bundle, ["t2"])
    truth = temporal_pair[1].truth
    assert iou(out.mask.array, ~truth["water"] & ~truth["builtup"]) > 0.95


# ------------------------------------------------------------------ SAR


def test_sar_water_otsu(registry, sar_bundle, sar_db) -> None:
    out = run_tool(registry.get("sar_water_otsu"), sar_bundle, ["sar"])
    assert iou(out.mask.array, sar_db.truth["water"]) > 0.9
    assert -22 < out.values["threshold_db"] < -8
    assert out.values["polarisation"] == "vv"
    assert out.confidence > 0.5
    assert not out.warnings


def test_sar_water_without_filter_is_worse(registry, sar_bundle, sar_db) -> None:
    raw = run_tool(
        registry.get("sar_water_otsu"),
        sar_bundle,
        ["sar"],
        params={"speckle_filter": "none", "min_region_px": 0},
    )
    filtered = run_tool(registry.get("sar_water_otsu"), sar_bundle, ["sar"])
    truth = sar_db.truth["water"]
    assert iou(filtered.mask.array, truth) > iou(raw.mask.array, truth)


def test_sar_water_implausible_threshold_warns(registry, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    land_only = 10 * np.log10(0.16 * rng.gamma(4, 0.25, (2, 64, 64))).astype(np.float32)
    path = write_array(tmp_path / "dry.tif", land_only, band_names=("VV", "VH"))
    out = run_tool(registry.get("sar_water_otsu"), validate_inputs([path]), ["sar"])
    assert out.warnings
    assert out.values["threshold_db"] == -12.0
    assert out.values["fraction"] < 0.05
    assert out.confidence < 0.5


# ------------------------------------------------------------------ fusion


def test_optsar_fusion(registry, optsar_bundle, optical_4band) -> None:
    opt = run_tool(registry.get("ndwi_water_mask"), optsar_bundle, ["optical"])
    sar = run_tool(registry.get("sar_water_otsu"), optsar_bundle, ["sar"])
    fused = run_tool(
        registry.get("optsar_fusion_rules"), optsar_bundle, upstream={"optical": opt, "sar": sar}
    )
    assert fused.values["agreement_iou"] > 0.85
    assert set(fused.masks) == {"mask", "agree", "optical_only", "sar_only"}
    assert iou(fused.masks["agree"].array, optical_4band.truth["water"]) > 0.85
    assert fused.confidence > 0.5
    inter = run_tool(
        registry.get("optsar_fusion_rules"),
        optsar_bundle,
        upstream={"optical": opt, "sar": sar},
        params={"rule": "intersection"},
    )
    assert inter.mask.array.sum() <= fused.mask.array.sum()


# ------------------------------------------------------------------ change


def test_change_cva(registry, temporal_bundle, temporal_pair) -> None:
    truth = temporal_pair[1].truth["builtup_gain"]
    out = run_tool(registry.get("change_cva"), temporal_bundle, ["t1", "t2"])
    assert iou(out.mask.array, truth) > 0.95
    assert out.values["normalisation"] == "linear"
    assert 1.0 < out.values["mean_gain"] < 1.1  # simulated 8 % brighter date 2
    assert out.confidence > 0.5


def test_cva_histogram_matching_failure_mode(registry, temporal_bundle, temporal_pair) -> None:
    """Documents why ``linear`` is the default: rank matching invents change over water."""
    truth = temporal_pair[1].truth
    hist = run_tool(
        registry.get("change_cva"),
        temporal_bundle,
        ["t1", "t2"],
        params={"normalisation": "histogram"},
    )
    linear = run_tool(registry.get("change_cva"), temporal_bundle, ["t1", "t2"])
    assert (hist.mask.array & truth["water"]).sum() > 100
    assert (linear.mask.array & truth["water"]).sum() == 0


def test_sar_logratio(registry, sar_temporal_bundle) -> None:
    bundle, truth = sar_temporal_bundle
    out = run_tool(registry.get("sar_logratio_change"), bundle, ["t1", "t2"])
    assert iou(out.mask.array, truth) > 0.9
    assert out.values["increase_pixels"] > 10 * out.values["decrease_pixels"]


def test_mask_compare_and_area(registry, temporal_bundle, temporal_pair) -> None:
    before = run_tool(registry.get("ndbi_builtup_mask"), temporal_bundle, ["t1"])
    after = run_tool(registry.get("ndbi_builtup_mask"), temporal_bundle, ["t2"])
    cmp = run_tool(
        registry.get("mask_compare"), temporal_bundle, upstream={"before": before, "after": after}
    )
    truth = temporal_pair[1]
    assert iou(cmp.masks["gain"].array, truth.truth["builtup_gain"]) > 0.95
    assert cmp.values["loss_pixels"] == 0
    assert cmp.values["relative_change"] is None  # nothing built-up before

    stats = run_tool(
        registry.get("area_stats"),
        temporal_bundle,
        upstream={"source": cmp},
        params={"masks": ["gain", "loss"]},
    )
    assert stats.values["gain_hectares"] == pytest.approx(
        truth.truth_hectares("builtup_gain"), rel=0.03
    )
    assert stats.values["loss_hectares"] == 0
    assert stats.values["gain_regions"] == 1
    assert stats.values["pixel_size_m"] == 10.0


def test_area_stats_geographic_crs(registry, synth_dir, optical_4band) -> None:
    """Geodesic areas: the truth mask warped (nearest) to EPSG:4326 keeps its hectares."""
    path = write_reprojected(optical_4band.path, synth_dir / "optical_wgs84.tif", "EPSG:4326")
    bundle = validate_inputs([path])
    ref = bundle.reference
    warped = np.zeros(ref.shape, dtype=np.uint8)
    reproject(
        optical_4band.truth["water"].astype(np.uint8),
        warped,
        src_transform=optical_4band.geometry.transform,
        src_crs="EPSG:32643",
        dst_transform=ref.affine,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
    )
    source = ToolOutput(masks={"mask": ref.georaster(warped.astype(bool))})
    stats = run_tool(registry.get("area_stats"), bundle, upstream={"source": source})
    assert stats.values["mask_hectares"] == pytest.approx(
        optical_4band.truth_hectares("water"), rel=0.02
    )


def test_area_stats_pixels_only_without_crs(registry, png_rgb) -> None:
    bundle = validate_inputs([png_rgb.path], benchmark=True)
    with pytest.raises(SatQueryError) as exc:  # RGB only: no NIR for NDWI
        run_tool(registry.get("ndwi_water_mask"), bundle, ["optical"])
    assert exc.value.code is ErrorCode.BAND_ROLE_MISSING

    truth = png_rgb.truth["water"]
    source = ToolOutput(masks={"mask": bundle.reference.georaster(truth)})
    stats = run_tool(registry.get("area_stats"), bundle, upstream={"source": source})
    assert stats.values["mask_pixels"] == truth.sum()
    assert stats.values["mask_hectares"] is None
    assert stats.values["pixel_size_m"] is None
    assert any("pixels only" in w for w in stats.warnings)


# ------------------------------------------------------------------ mocks


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("Is there a river?", "yes"),
        ("How many buildings are there?", "0"),
        ("What is the main land cover?", "unknown"),
        (None, "unknown"),
    ],
)
def test_prior_answer(question, answer) -> None:
    assert prior_answer(question) == answer


def test_mock_tools_are_labelled(registry, opt_bundle, temporal_bundle) -> None:
    vqa = run_tool(registry.get("vlm_vqa"), opt_bundle, ["optical"], query="Is there water?")
    cap = run_tool(registry.get("vlm_caption"), opt_bundle, ["optical"])
    box = run_tool(registry.get("grounding"), opt_bundle, ["optical"], query="the lake")
    chg = run_tool(registry.get("change_vqa"), temporal_bundle, ["t1", "t2"], query="Any change?")
    for out in (vqa, cap, box, chg):
        assert out.is_mock
        assert out.text.startswith(MOCK_PREFIX)
        assert out.confidence == 0.0
    assert vqa.values["answer"] == "yes"
    assert box.boxes[0].bbox_px == (0, 0, 128, 128)
    assert "2020-03-01 -> 2024-03-01" in chg.text
