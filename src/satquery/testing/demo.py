"""Write the synthetic demo scenes to a folder: ``python -m satquery.testing.demo [out_dir]``."""

from __future__ import annotations

import sys
from pathlib import Path

from satquery.testing.synthetic import write_optical, write_sar, write_temporal_pair

QUERIES = {
    "optical_4band.tif": "How much area is covered by water?",
    "sar_vv_vh.tif": "Is there any water in this image?",
    "optical_4band.tif + sar_vv_vh.tif": "Map the water using optical and SAR",
    "temporal/s2_20200301.tif + temporal/s2_20240301.tif": "Has built-up area increased?",
}


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_optical(out_dir / "optical_4band.tif")
    write_sar(out_dir / "sar_vv_vh.tif")
    write_temporal_pair(out_dir / "temporal")
    print(f"Demo scenes written to {out_dir.resolve()}\nTry:")
    for files, query in QUERIES.items():
        print(f"  {files:<55} {query}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demo"))
