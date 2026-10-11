"""BigEarthNet.txt: text annotations (parquet) over BigEarthNet v2.0 Sentinel-1/2 patches.

Columns used (as read by the dataset's own ``ben_txt_datamodule.py``): ``patch_id``,
``s1_name``, ``input``, ``output``, ``type``, ``split`` (train / validation / test /
bench). Images are NOT part of the dataset: pass ``image_root`` pointing at a local
BigEarthNet v2.0 Sentinel-2 folder.

TODO(confirm): the per-band file naming inside a patch folder (``<patch>_B02.tif`` ...)
follows the BigEarthNet v2 archive as we understand it; verify against a real copy.
Bands are stacked to a 10 m, 6-band GeoTIFF (B, G, R, NIR, SWIR1, SWIR2) cached under
``<root>/stacked``. Rows whose ``type`` mentions grounding are skipped (box format not
yet verified).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from satquery.eval.datasets.base import BenchmarkDataset, BenchmarkSample, DatasetNotFoundError
from satquery.types import Modality, TaskType

PARQUET = "BigEarthNet.txt.parquet"
S2_BANDS = (("B02", "blue"), ("B03", "green"), ("B04", "red"), ("B08", "nir"),
            ("B11", "swir1"), ("B12", "swir2"))  # fmt: skip


def task_for(type_value: str) -> TaskType | None:
    t = type_value.lower()
    if "caption" in t:
        return TaskType.CAPTION
    if any(k in t for k in ("ground", "refer", "bbox", "box")):
        return None
    return TaskType.VQA


class BigEarthNetTxt(BenchmarkDataset):
    name = "bigearthnet-txt"
    tasks = frozenset({TaskType.VQA, TaskType.CAPTION})
    modality = Modality.MULTISPECTRAL

    def __init__(self, root: Path, split: str = "bench", image_root: Path | None = None) -> None:
        super().__init__(root, split)
        self.image_root = image_root

    def _stack(self, patch_id: str) -> Path | None:
        if self.image_root is None:
            return None
        out = self.root / "stacked" / f"{patch_id}.tif"
        if out.exists():
            return out
        folders = [p for p in self.image_root.rglob(patch_id) if p.is_dir()]
        if not folders:
            return None
        files = [folders[0] / f"{patch_id}_{band}.tif" for band, _ in S2_BANDS]
        if not all(f.exists() for f in files):
            return None
        with rasterio.open(files[0]) as ref:
            profile = ref.profile | {"count": len(files), "dtype": "uint16", "driver": "GTiff"}
            shape = (ref.height, ref.width)
        stack = np.empty((len(files), *shape), dtype=np.uint16)
        for i, f in enumerate(files):
            with rasterio.open(f) as src:
                stack[i] = src.read(1, out_shape=shape, resampling=Resampling.bilinear)
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(stack)
            for i, (_, role) in enumerate(S2_BANDS, start=1):
                dst.set_band_description(i, role)
        return out

    def samples(self) -> Iterator[BenchmarkSample]:
        path = self.root / PARQUET
        self.require(path)
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the [eval] extra
            raise DatasetNotFoundError("BigEarthNet.txt needs the [eval] extra (pandas).") from exc
        columns = ["patch_id", "input", "output", "type", "split"]
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["split"] == self.split]
        for idx, row in frame.iterrows():
            task = task_for(str(row["type"]))
            if task is None:
                continue
            image = self._stack(str(row["patch_id"]))
            if image is None:
                continue
            yield BenchmarkSample(
                id=f"{self.name}:{self.split}:{idx}",
                task=task,
                images=[image],
                question=str(row["input"]),
                question_type=str(row["type"]),
                answers=[str(row["output"])],
                meta={"patch_id": row["patch_id"]},
            )
