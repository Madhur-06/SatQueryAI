"""Where each benchmark comes from. Every entry was checked against the primary source on
2026-09-29; nothing here is downloaded automatically.
"""

from __future__ import annotations

from pydantic import BaseModel


class DatasetSource(BaseModel):
    name: str
    homepage: str
    files: list[str]
    license: str
    notes: str


SOURCES: dict[str, DatasetSource] = {
    "rsvqa-lr": DatasetSource(
        name="RSVQA Low Resolution (Lobry et al., 2020)",
        homepage="https://zenodo.org/records/6344334 (concept DOI 10.5281/zenodo.6344333)",
        files=[
            "Images_LR.zip (95.0 MB) -> Images_LR/<image id>.tif",
            "LR_split_{train,val,test}_{questions,answers,images}.json",
        ],
        license="CC-BY-4.0",
        notes="772 Sentinel-2 RGB tiles (256 x 256, 10 m). JSON schema as read by the "
        "authors' VQA_model/VQALoader.py (github.com/syvlo/RSVQA).",
    ),
    "vrsbench": DatasetSource(
        name="VRSBench (Li et al., NeurIPS 2024 D&B)",
        homepage="https://huggingface.co/datasets/xiang709/VRSBench",
        files=[
            "Images_val.zip",
            "VRSBench_EVAL_vqa.json",
            "VRSBench_EVAL_Cap.json",
            "VRSBench_EVAL_referring.json",
        ],
        license="CC-BY-4.0",
        notes="Eval files are JSON lists of {image_id, question, ground_truth, type, "
        "question_id}; referring boxes are '{<x1><y1><x2><y2>}' on a 0-100 scale.",
    ),
    "cdvqa": DatasetSource(
        name="CDVQA (Yuan et al., TGRS 2022)",
        homepage="https://github.com/YZHJessica/CDVQA",
        files=["{Train,Val,Test,Test2}_{questions,answers,images}.json"],
        license="Apache-2.0 (annotations)",
        notes="Repository contains annotations only (RSVQA-style schema, images referenced "
        "by file_name). The bi-temporal images come from the SECOND dataset; the repo does "
        "not document where to obtain them. TODO: confirm image source with the team.",
    ),
    "bigearthnet-txt": DatasetSource(
        name="BigEarthNet.txt (BIFOLD)",
        homepage="https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
        files=["BigEarthNet.txt.parquet (text only, ~467 MB)"],
        license="CDLA-Permissive-1.0",
        notes="Columns patch_id, s1_name, input, output, type, category, split "
        "(train/validation/test/bench). Images must come from a local BigEarthNet v2.0 "
        "(reBEN) copy, tens of GB; never downloaded by SatQuery.",
    ),
}
