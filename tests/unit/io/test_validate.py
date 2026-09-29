from datetime import date
from pathlib import Path

import numpy as np
import pytest

from satquery.errors import ErrorCode, SatQueryError
from satquery.io.validate import InputOverrides, check_task_compatibility, validate_inputs
from satquery.testing.synthetic import SceneGeometry, write_array, write_optical
from satquery.types import BandRole, InputConfig, Modality, TaskType


def _error(fn, *args, **kwargs) -> ErrorCode:
    with pytest.raises(SatQueryError) as exc:
        fn(*args, **kwargs)
    return exc.value.code


# ------------------------------------------------------------------ single images


def test_single_optical(optical_4band) -> None:
    bundle = validate_inputs([optical_4band.path])
    assert bundle.config is InputConfig.SINGLE_OPT
    assert list(bundle.layers) == ["optical"]
    layer = bundle.reference
    assert layer.modality is Modality.OPTICAL
    assert layer.modality_source == "detected"
    assert layer.units == "as_stored"
    assert layer.band(BandRole.NIR).shape == (128, 128)
    assert bundle.alignment is None
    assert bundle.warnings == []


def test_single_sar_linear_is_converted_to_db(sar_db, sar_linear) -> None:
    db = validate_inputs([sar_db.path]).layer("sar")
    lin = validate_inputs([sar_linear.path]).layer("sar")
    assert lin.units == db.units == "db"
    assert lin.sar_input_scale == "linear"
    assert db.sar_input_scale == "db"
    np.testing.assert_allclose(lin.data, db.data, atol=1e-3)  # same seed => same scene


def test_modality_override(sar_db) -> None:
    bundle = validate_inputs([sar_db.path], [InputOverrides(modality=Modality.OPTICAL)])
    assert bundle.config is InputConfig.SINGLE_OPT
    assert bundle.reference.modality_source == "override"


def test_missing_band_role_is_structured(optical_4band) -> None:
    layer = validate_inputs([optical_4band.path]).reference
    code = _error(layer.band, BandRole.SWIR1)
    assert code is ErrorCode.BAND_ROLE_MISSING


def test_unknown_modality_rejected(tmp_path: Path) -> None:
    data = 0.5 + np.random.default_rng(0).normal(0, 0.001, (2, 32, 32)).astype(np.float32)
    path = write_array(tmp_path / "mystery.tif", data)
    assert _error(validate_inputs, [path]) is ErrorCode.MODALITY_UNKNOWN


def test_benchmark_png(png_rgb) -> None:
    assert _error(validate_inputs, [png_rgb.path]) is ErrorCode.UNSUPPORTED_FORMAT
    bundle = validate_inputs([png_rgb.path], benchmark=True)
    assert bundle.config is InputConfig.SINGLE_OPT
    assert bundle.reference.crs is None
    assert any("no CRS" in w for w in bundle.warnings)


def test_input_count_errors(optical_4band, sar_db, optical_6band) -> None:
    assert _error(validate_inputs, []) is ErrorCode.NO_INPUTS
    paths = [optical_4band.path, sar_db.path, optical_6band.path]
    assert _error(validate_inputs, paths) is ErrorCode.TOO_MANY_INPUTS
    code = _error(validate_inputs, [optical_4band.path], [None, None])
    assert code is ErrorCode.INVALID_OVERRIDE


# ------------------------------------------------------------------ pairs


@pytest.mark.parametrize("sar_first", [False, True])
def test_opt_sar_pair(optical_4band, sar_db, sar_first: bool) -> None:
    paths = [sar_db.path, optical_4band.path] if sar_first else [optical_4band.path, sar_db.path]
    bundle = validate_inputs(paths)
    assert bundle.config is InputConfig.PAIR_OPT_SAR
    assert list(bundle.layers) == ["optical", "sar"]
    assert bundle.alignment is not None
    assert bundle.alignment.crs_match
    assert bundle.alignment.footprint_iou == pytest.approx(1.0)
    assert not bundle.alignment.resampled


def test_temporal_pair_sorted_by_date(temporal_pair) -> None:
    t1, t2 = temporal_pair
    bundle = validate_inputs([t2.path, t1.path])  # uploaded newest first
    assert bundle.config is InputConfig.PAIR_TEMPORAL
    assert bundle.layer("t1").acquired == date(2020, 3, 1)
    assert bundle.layer("t2").acquired == date(2024, 3, 1)
    assert bundle.layer("t1").meta.filename == t1.path.name


def test_temporal_date_override_changes_order(temporal_pair) -> None:
    t1, t2 = temporal_pair
    overrides = [InputOverrides(acquired=date(2030, 1, 1)), None]
    bundle = validate_inputs([t1.path, t2.path], overrides)
    assert bundle.layer("t1").meta.filename == t2.path.name
    assert bundle.layer("t2").acquired_source == "override"


def test_reprojection_onto_reference_grid(optical_4band, sar_other_crs) -> None:
    bundle = validate_inputs([optical_4band.path, sar_other_crs])
    assert bundle.alignment is not None
    assert not bundle.alignment.crs_match
    assert bundle.alignment.resampled
    # The warped file's bounding box is larger than the scene (rotated grid), so IoU < 1.
    assert bundle.alignment.footprint_iou > 0.8
    sar = bundle.layer("sar")
    assert sar.crs == "EPSG:32643"
    assert sar.shape == (128, 128)
    assert any("reprojected" in w for w in bundle.warnings)
    # The water body still lands where it should after warping.
    water_truth = optical_4band.truth["water"]
    sar_water = (sar.band(BandRole.VV) < -15) & sar.valid
    inter = (sar_water & water_truth).sum()
    union = (sar_water | water_truth).sum()
    assert inter / union > 0.7


def test_crs_mismatch_rejected_when_reprojection_disallowed(optical_4band, sar_other_crs) -> None:
    code = _error(validate_inputs, [optical_4band.path, sar_other_crs], allow_reproject=False)
    assert code is ErrorCode.CRS_MISMATCH


def test_pair_without_crs_rejected(optical_4band, optical_no_crs) -> None:
    code = _error(validate_inputs, [optical_4band.path, optical_no_crs.path])
    assert code is ErrorCode.MISSING_CRS


def test_no_overlap(optical_4band, optical_disjoint) -> None:
    with pytest.raises(SatQueryError) as exc:
        validate_inputs([optical_4band.path, optical_disjoint.path])
    assert exc.value.code is ErrorCode.NO_OVERLAP
    assert exc.value.info.details["footprint_iou"] == 0.0


def test_low_overlap(tmp_path: Path, optical_4band) -> None:
    shifted = write_optical(tmp_path / "half.tif", geometry=SceneGeometry().shifted(900, 0))
    code = _error(validate_inputs, [optical_4band.path, shifted.path])
    assert code is ErrorCode.LOW_OVERLAP
    # ...but accepted with a lower threshold, and resampled onto the reference grid.
    bundle = validate_inputs([optical_4band.path, shifted.path], min_overlap_iou=0.1)
    assert bundle.alignment is not None
    assert bundle.alignment.resampled
    assert not bundle.layer("t2").valid[:, :50].any()  # outside the second footprint


def test_incompatible_temporal_bands(tmp_path: Path) -> None:
    a = write_optical(tmp_path / "a_20200101.tif", n_bands=4, band_descriptions=False)
    b = write_optical(tmp_path / "b_20210101.tif", n_bands=6, band_descriptions=False)
    assert _error(validate_inputs, [a.path, b.path]) is ErrorCode.INCOMPATIBLE_PAIR


def test_missing_dates_warn_and_keep_upload_order(tmp_path: Path) -> None:
    a = write_optical(tmp_path / "a.tif", acquired=None)
    b = write_optical(tmp_path / "b.tif", acquired=None, seed=5)
    bundle = validate_inputs([a.path, b.path])
    assert bundle.layer("t1").meta.filename == "a.tif"
    assert any("upload order" in w for w in bundle.warnings)


# ------------------------------------------------------------------ task vs configuration


@pytest.mark.parametrize(
    ("task", "config", "ok"),
    [
        (TaskType.CHANGE_VQA, InputConfig.SINGLE_OPT, False),
        (TaskType.CHANGE_DESC, InputConfig.PAIR_OPT_SAR, False),
        (TaskType.CHANGE_VQA, InputConfig.PAIR_TEMPORAL, True),
        (TaskType.FUSION_EXTRACT, InputConfig.PAIR_TEMPORAL, False),
        (TaskType.FUSION_EXTRACT, InputConfig.PAIR_OPT_SAR, True),
        (TaskType.VQA, InputConfig.SINGLE_SAR, True),
        (TaskType.CAPTION, InputConfig.PAIR_TEMPORAL, True),
    ],
)
def test_task_compatibility(task: TaskType, config: InputConfig, ok: bool) -> None:
    if ok:
        check_task_compatibility(task, config)
    else:
        assert _error(check_task_compatibility, task, config) is ErrorCode.TASK_INPUT_MISMATCH


def test_change_query_with_one_image_message() -> None:
    with pytest.raises(SatQueryError) as exc:
        check_task_compatibility(TaskType.CHANGE_VQA, InputConfig.SINGLE_OPT)
    assert "two images" in exc.value.info.message
    assert exc.value.info.details["supported_configs"] == [InputConfig.PAIR_TEMPORAL]
