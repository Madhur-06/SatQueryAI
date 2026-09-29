# PROGRESS

Phase 1 plan: walking skeleton, no training. Milestones M1–M6. Tooling: uv + just (Windows-friendly).

## Done
### M1: Project scaffold (2026-09-29)
- uv project, src layout (`src/satquery/`), hatchling build, Python 3.12.
- Extras: `[ml]` (torch, transformers ≥ 4.57, accelerate) and `[eval]` (pycocoevalcap, huggingface_hub, pyarrow, pandas). Dev group: pytest, pytest-cov, ruff, pre-commit, httpx.
- `justfile`: `install`, `install-eval`, `install-ml`, `lint`, `fmt`, `test`, `serve`. PowerShell on Windows, sh elsewhere.
- pre-commit: pre-commit-hooks v6.0.0; ruff check + format run through `uv run`, so the version is pinned by `uv.lock`.
- `satquery.config.Settings` (env prefix `SATQUERY_`). The VLM is off by default (`vlm_enabled=False`); model id `Qwen/Qwen3-VL-2B-Instruct`.
- Minimal FastAPI app with `GET /health`, so `just serve` already works.
- `satquery.testing.synthetic`: generators with ground-truth masks, all on a 128×128, 10 m grid in EPSG:32643.
  - 4-band and 6-band optical (uint16 reflectance, water disk).
  - VV/VH SAR in dB or linear, with gamma speckle.
  - 6-band temporal pair where a built-up block appears in 2024, with dates in both tags and filenames.
  - Reprojected (EPSG:32644), no-CRS, disjoint-footprint and PNG variants for failure tests.
- `tests/conftest.py` session fixtures + fixture sanity tests (NDWI/NDBI/backscatter separability).

### M2: Input validation & preprocessing, `satquery/io/` (2026-09-29)
- `types.py` (shared enums, canonical layer keys) and `errors.py` (`SatQueryError` + `ErrorCode` + `ErrorInfo`).
- `raster.inspect_raster`: driver/suffix checks (PNG/JPEG only with `benchmark=True`), CRS as `EPSG:x`, transform, res, bounds, nodata, band descriptions, colour interp, dataset and band tags, sha256.
  - Acquisition date comes from tags first, then the filename (`dates.py`).
- `modality.detect_modality`. Evidence order: band descriptions > sensor tags > statistics (dB range, speckle local-CV, band count). Returns a confidence and reasons. ≥ 5 optical bands → MULTISPECTRAL.
- `bands.resolve_bands`. Order: override (1-based) > descriptions > wavelength tags > RGB colour interp > sensor code tables (only when the sensor is named: S2 `B8` = NIR, Landsat `B8` = PAN).
  - There is no positional default.
- `validate.validate_inputs` → `InputBundle` (SINGLE_OPT / SINGLE_SAR / PAIR_OPT_SAR / PAIR_TEMPORAL).
  - Temporal pairs are ordered by date. Warnings: opt–SAR date gap, missing dates.
  - Pairs: CRS check (reproject by default; `CRS_MISMATCH` if `allow_reproject=False`), `MISSING_CRS`, footprint IoU (`NO_OVERLAP` / `LOW_OVERLAP`), and resampling onto the reference grid.
  - SAR is always stored in dB and warped in linear power.
- `check_task_compatibility` covers change-with-one-image, fusion without an opt–SAR pair, etc.
- Preprocessing:
  - SAR: scale detection, dB↔linear, Lee filter with ENL estimate, VV/VH/ratio composite.
  - Optical: percentile stretch, true/false colour, normalized difference.
  - Temporal: histogram matching.
- `tiling`: overlapping tiles with per-tile geotransform; centre-crop `stitch`; `run_tiled`.
- `geometry`: bounds IoU, semantic CRS equality, per-row geodesic pixel area for geographic CRSs.
- `vector.mask_to_geojson`: traced in native CRS, reprojected to WGS84 (RFC 7946); pixel-space fallback.
- `render`: quicklooks, RGBA mask layers, blends, PNG writer.
- 90 tests pass, including every failure case in the spec.

## Next
- M3: `satquery/tools/`: Tool base, Registry, classical tools, mocks, lazy Qwen3-VL tool.

## Open questions / decisions log
- Starlette warns that `httpx` for its TestClient is deprecated (suggests `httpx2`). Harmless for now; revisit once the replacement is verified.
- Whole rasters are read into memory in Phase 1. Tiling utilities exist, but tools use them only above `tile_size`.
- Lee filter only. Refined Lee is a TODO (Phase 2).
- Built-up needs SWIR (NDBI). 4-band inputs will get a structured `BAND_ROLE_MISSING` error (decided).
- Old pip `venv/` in the repo root is unused (uv uses `.venv/`); it's git-ignored and safe to delete.
- M6: confirm RSVQA-LR download URLs and the CDVQA image hosting before writing download code.
