import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from satquery.eval.cli import main
from satquery.eval.datasets import DatasetNotFoundError, load_dataset
from satquery.eval.datasets.vrsbench import parse_box
from satquery.eval.metrics import (
    box_iou,
    change_mask_scores,
    grounding_scores,
    normalize_answer,
    vqa_accuracy,
)
from satquery.eval.report import to_markdown, write_report
from satquery.eval.runner import run_eval
from satquery.types import TaskType

needs_coco = pytest.mark.skipif(
    importlib.util.find_spec("pycocoevalcap") is None, reason="needs the [eval] extra"
)

# ------------------------------------------------------------------ metrics


@pytest.mark.parametrize(
    ("raw", "norm"),
    [(True, "yes"), (False, "no"), ("Yes.", "yes"), (" 12 ", "12"), (3.0, "3"), ("12.0", "12"),
     ("The Harbour", "harbour"), ("between 10m2 and 100m2", "between 10m2 and 100m2")],
)  # fmt: skip
def test_normalize_answer(raw, norm) -> None:
    assert normalize_answer(raw) == norm


def test_vqa_accuracy_overall_and_per_type() -> None:
    records = [("yes", ["yes"], "presence"), ("no", ["yes"], "presence"),
               ("3", ["3"], "count"), (None, ["1"], "count"), ("Yes", ["yes"], "comp")]  # fmt: skip
    m = vqa_accuracy(records)
    assert m["n"] == 5
    assert m["overall_accuracy"] == pytest.approx(3 / 5)
    assert m["per_type"]["presence"] == {"accuracy": 0.5, "n": 2}
    assert m["average_accuracy"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_grounding_metrics() -> None:
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    m = grounding_scores([((0, 0, 10, 10), (0, 0, 10, 10)), ((0, 0, 10, 10), (5, 0, 15, 10)),
                          (None, (0, 0, 1, 1))])  # fmt: skip
    assert m["acc@0.5"] == pytest.approx(1 / 3)
    assert m["mean_iou"] == pytest.approx((1 + 1 / 3) / 3)


def test_change_mask_metrics() -> None:
    gt = np.zeros((4, 4), bool)
    gt[:2] = True  # 8 px
    pred = np.zeros((4, 4), bool)
    pred[:1] = True  # 4 px, all correct
    m = change_mask_scores([(pred, gt)])
    assert m["micro"]["precision"] == 1.0
    assert m["micro"]["recall"] == 0.5
    assert m["micro"]["f1"] == pytest.approx(2 / 3)
    assert m["micro"]["iou"] == 0.5
    with pytest.raises(ValueError, match="shape"):
        change_mask_scores([(pred, gt[:2])])


@needs_coco
def test_caption_metrics() -> None:
    from satquery.eval.metrics import caption_scores

    refs = {"1": ["a harbour with two ships docked near a pier"], "2": ["a dense urban area"]}
    perfect = caption_scores({"1": refs["1"][0], "2": refs["2"][0]}, refs, meteor=False)
    poor = caption_scores({"1": "a forest", "2": "water"}, refs, meteor=False)
    assert perfect["bleu4"] > 0.9
    assert perfect["cider"] > poor["cider"]
    assert poor["bleu4"] < 0.1


def test_parse_vrsbench_box() -> None:
    assert parse_box("{<25><40><33><60>}") == (25, 40, 33, 60)
    assert parse_box("no box") is None


# ------------------------------------------------------------------ loaders


def test_rsvqa_loader(rsvqa_root: Path) -> None:
    ds = load_dataset("rsvqa-lr", rsvqa_root)
    samples = list(ds.samples())
    assert len(samples) == 3  # the inactive question is skipped
    assert samples[0].images == [rsvqa_root / "Images_LR" / "0.tif"]
    assert samples[1].answers == ["0"]
    assert samples[1].question_type == "count"
    assert ds.subset(2, seed=1) == ds.subset(2, seed=1)
    assert len(ds.subset(99)) == 3


def test_cdvqa_loader(cdvqa_root: Path) -> None:
    samples = list(load_dataset("cdvqa", cdvqa_root).samples())
    assert samples[0].task is TaskType.CHANGE_VQA
    assert samples[0].images == [cdvqa_root / "im1" / "00000.png", cdvqa_root / "im2" / "00000.png"]


def test_vrsbench_loaders(vrsbench_root: Path) -> None:
    vqa = list(load_dataset("vrsbench-vqa", vrsbench_root).samples())
    assert len(vqa) == 2  # sample with a missing image is skipped
    assert vqa[0].images[0].name == "P0001_0001.png"  # found in a nested folder
    ref = list(load_dataset("vrsbench-grounding", vrsbench_root).samples())
    assert ref[1].bbox == (60, 40, 80, 60)
    cap = list(load_dataset("vrsbench-caption", vrsbench_root).samples())
    assert cap[0].answers[0].startswith("A harbour")


def test_missing_dataset_explains_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match=r"zenodo\.org/records/6344334"):
        list(load_dataset("rsvqa-lr", tmp_path).samples())


# ------------------------------------------------------------------ runner / report / CLI


def test_run_eval_mock_vqa(rsvqa_root: Path, tmp_path: Path) -> None:
    report = run_eval(load_dataset("rsvqa-lr", rsvqa_root), "vlm_vqa")
    assert report.is_mock
    assert report.n_samples == 3
    assert report.n_errors == 0
    # Mock prior: "yes" for yes/no, "0" for counts -> 2 of 3 correct here.
    assert report.metrics["overall_accuracy"] == pytest.approx(2 / 3)
    json_path, md_path = write_report(report, tmp_path / "results")
    assert json.loads(json_path.read_text())["benchmark"] == "rsvqa-lr"
    md = md_path.read_text(encoding="utf-8")
    assert "Mock tool" in md
    assert "Per question type" in md


def test_run_eval_grounding_and_change(vrsbench_root: Path, cdvqa_root: Path) -> None:
    ground = run_eval(load_dataset("vrsbench-grounding", vrsbench_root), "grounding")
    assert ground.records[0].bbox_pred == (0, 0, 100, 100)
    assert ground.metrics["acc@0.5"] == 0.5  # full-image mock box matches only box 1
    change = run_eval(load_dataset("cdvqa", cdvqa_root), "change_vqa")
    assert change.n_errors == 0
    assert change.metrics["n"] == 3


def test_run_eval_pipeline_mode(rsvqa_root: Path) -> None:
    report = run_eval(load_dataset("rsvqa-lr", rsvqa_root), "pipeline")
    assert report.mode == "pipeline"
    assert report.n_samples == 3
    assert "overall_accuracy" in report.metrics
    assert to_markdown(report)


def test_run_eval_rejects_incompatible_tool(rsvqa_root: Path) -> None:
    with pytest.raises(ValueError, match="does not support"):
        run_eval(load_dataset("rsvqa-lr", rsvqa_root), "ndwi_water_mask")


def test_cli(rsvqa_root: Path, tmp_path: Path, capsys) -> None:
    assert main(["--list"]) == 0
    assert "rsvqa-lr" in capsys.readouterr().out
    out = tmp_path / "results"
    code = main(["--benchmark", "rsvqa-lr", "--tool", "vlm_vqa", "--subset", "2",
                 "--data-root", str(rsvqa_root), "--out", str(out)])  # fmt: skip
    assert code == 0
    assert len(list(out.glob("*.json"))) == 1
    assert len(list(out.glob("*.md"))) == 1
    missing = main(["--benchmark", "rsvqa-lr", "--tool", "vlm_vqa",
                    "--data-root", str(tmp_path / "nope")])  # fmt: skip
    assert missing == 2
    assert "zenodo" in capsys.readouterr().err
