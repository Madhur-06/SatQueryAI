"""Contact sheet of random patches x composites, plus per-channel code statistics.

    PYTHONPATH=src python scripts/preview_composites.py --mirror local:data/benvqa/dev_hub \
        --split train --n 48 --out reports/previews/train.png

Each row is one patch: rgb | nir_swir | red_edge | sar, then the country / season label.
The JSON next to the PNG reports, per channel, the share of codes at 0 and 255
(saturation). Use it to tune ``max_b`` (quantization) and ``white`` (display).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from benvqa.data.composites import DEFAULT_COMPOSITES, make_composite
from benvqa.data.quantize import CHANNELS, QuantSpec, cube_from_bytes
from benvqa.hub_sync import make_mirror

TILE = 120


def load_rows(mirror_spec: str, split: str, n: int, seed: int, cache: Path) -> list[dict]:
    mirror = make_mirror(mirror_spec)
    shards = [p for p in mirror.list(f"{split}/") if p.endswith(".parquet")]
    if not shards:
        raise SystemExit(f"no shards under {split}/ on {mirror.describe()}")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for rel in rng.permutation(shards)[:4]:
        table = pq.read_table(mirror.download(str(rel), cache))
        rows += table.select(["patch_id", "country", "season", "has_s1", "bands"]).to_pylist()
    take = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in take]


def contact_sheet(
    rows: list[dict], spec: QuantSpec, names: list[str], scale: int = 1
) -> Image.Image:
    t = TILE * scale
    label_w = 150
    per_row = 2  # patches per sheet row
    cell_w = len(names) * t + label_w
    n_rows = -(-len(rows) // per_row)
    sheet = Image.new("RGB", (per_row * cell_w, n_rows * (t + 4) + 18), "white")
    draw = ImageDraw.Draw(sheet)
    for j in range(per_row):
        for k, name in enumerate(names):
            draw.text((j * cell_w + k * t + 4, 2), name, fill="black")
    for i, r in enumerate(rows):
        cube = cube_from_bytes(r["bands"])
        x0, y0 = (i % per_row) * cell_w, 18 + (i // per_row) * (t + 4)
        for k, name in enumerate(names):
            img = Image.fromarray(make_composite(cube, DEFAULT_COMPOSITES[name], spec))
            sheet.paste(img.resize((t, t), Image.Resampling.NEAREST), (x0 + k * t, y0))
        label = f"{r['country']}\n{r['season']}" + ("" if r["has_s1"] else "\n(no S1)")
        draw.text((x0 + len(names) * t + 4, y0 + 4), label, fill="black")
    return sheet


def channel_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    cubes = np.stack([cube_from_bytes(r["bands"]) for r in rows])
    out = {}
    for c, name in enumerate(CHANNELS):
        x = cubes[:, c]
        out[name] = {
            "p_zero": float((x == 0).mean()),
            "p_255": float((x == 255).mean()),
            "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--composites", default="rgb,nir_swir,red_edge,sar")
    ap.add_argument("--out", type=Path, default=Path("reports/previews/train.png"))
    args = ap.parse_args()
    spec = QuantSpec()
    with tempfile.TemporaryDirectory() as tmp:
        rows = load_rows(args.mirror, args.split, args.n, args.seed, Path(tmp))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(rows, spec, args.composites.split(","), args.scale).save(args.out)
    stats = channel_stats(rows)
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"wrote {args.out} ({len(rows)} patches)")
    for name, s in stats.items():
        pct = f"zero {s['p_zero']:.3%}  sat {s['p_255']:.3%}"
        print(f"  {name:4s} {pct}  p50 {s['p50']:.0f}  p99 {s['p99']:.0f}")


if __name__ == "__main__":
    main()
