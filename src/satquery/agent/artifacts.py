"""Write evidence files for a request: quicklooks, mask layers, overlays, GeoTIFF, GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from satquery.agent.composer import FinalAnswer
from satquery.agent.executor import StepResult
from satquery.agent.trace import Artifact
from satquery.io.render import blend, colour_for, mask_rgba, quicklook, save_png
from satquery.io.validate import InputBundle
from satquery.io.vector import mask_to_geojson


def write_artifacts(
    out_dir: Path, bundle: InputBundle, results: list[StepResult], answer: FinalAnswer | None
) -> list[Artifact]:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Artifact] = []
    looks: dict[str, np.ndarray] = {}
    for key, layer in bundle.layers.items():
        looks[key] = quicklook(layer.data, layer.valid, layer.modality, layer.band_map)
        name = f"quicklook_{key}.png"
        save_png(looks[key], out_dir / name)
        artifacts.append(
            Artifact(kind="quicklook", label=f"{key} ({layer.meta.filename})", filename=name,
                     media_type="image/png", layer=key)
        )  # fmt: skip
    if answer is None:
        return artifacts

    outputs = {r.step_id: r.output for r in results if r.output is not None}
    for ev in answer.evidence:
        out = outputs.get(ev.step_id)
        if ev.kind != "mask" or out is None or ev.key not in out.masks:
            continue
        grid = out.masks[ev.key]
        mask = grid.array.astype(bool)
        colour = colour_for(f"{ev.key} {ev.label}")
        base = ev.step_id if ev.key == "mask" else f"{ev.step_id}_{ev.key}"
        common = {"label": ev.label, "layer": ev.layer, "step_id": ev.step_id}

        save_png(mask_rgba(mask, colour, alpha=255), out_dir / f"{base}_mask.png")
        artifacts.append(
            Artifact(kind="mask_png", filename=f"{base}_mask.png", media_type="image/png", **common)
        )
        look = looks.get(ev.layer, next(iter(looks.values())))
        save_png(blend(look, mask, colour, alpha=0.55), out_dir / f"{base}_overlay.png")
        artifacts.append(
            Artifact(kind="overlay", filename=f"{base}_overlay.png", media_type="image/png",
                     **common)
        )  # fmt: skip
        grid.with_array(mask.astype(np.uint8)).to_geotiff(out_dir / f"{base}.tif")
        artifacts.append(
            Artifact(kind="geotiff", filename=f"{base}.tif", media_type="image/tiff", **common)
        )
        geojson = mask_to_geojson(grid.with_array(mask), properties={"label": ev.label})
        (out_dir / f"{base}.geojson").write_text(json.dumps(geojson), encoding="utf-8")
        artifacts.append(
            Artifact(kind="geojson", filename=f"{base}.geojson",
                     media_type="application/geo+json", **common)
        )  # fmt: skip
    return artifacts
