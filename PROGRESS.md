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

### M3: Tool registry & tools, `satquery/tools/` (2026-09-29)
- `Tool` ABC (`base.py`). Class-level `name`, `version`, `tasks`, `input_configs`, `produces` (Capability), `requires` (layer count, modality family, band-role alternatives, named upstream inputs), and `Params` (pydantic, `extra="forbid"`, bounded).
  - `invoke()` checks availability → params → requirements → upstream, then runs and stamps the runtime and mock flag.
  - Errors: `INVALID_PARAMS`, `MODALITY_MISMATCH`, `BAND_ROLE_MISSING`, `TOOL_UNAVAILABLE`.
- `ToolOutput`: text, boolean `masks` (`"mask"` = primary), continuous `rasters`, `geojson`, `boxes`, scalar `values`, `confidence` + `confidence_basis`, runtime, `is_mock`, warnings. `summary()` is JSON-safe for the trace.
- `Registry`: register, get, `find(task, config, capability, layers)` ranked available > real > priority, `export_json()` (full param JSON schemas) and `digest()`.
- Classical tools, each with a documented confidence heuristic:
  - `ndwi_water_mask` (MNDWI automatically when SWIR is present), `ndbi_builtup_mask` (excludes NDVI > 0.3 and water), `ndvi_vegetation_mask`.
  - `sar_water_otsu`: Lee filter, Otsu in dB, plausibility cap at -12 dB with a warning.
  - `optsar_fusion_rules`: union/intersection; agree/optical-only/SAR-only masks; agreement IoU feeds confidence.
  - `change_cva`: common bands, robust linear normalisation, standardised magnitude, Otsu.
  - `sar_logratio_change`: increase/decrease masks.
  - `mask_compare`: gain/loss per class.
  - `area_stats`: every upstream mask → pixels, hectares (geodesic on geographic CRS), fraction, connected regions.
- Mocks `vlm_vqa`, `vlm_caption`, `grounding`, `change_vqa`: labelled `[MOCK]`, confidence 0, trivial prior answers ("yes" / "0") so eval gets a floor baseline.
- `qwen_vl_zeroshot` (priority over mocks). Unavailable unless `SATQUERY_VLM_ENABLED=true` and torch + transformers are installed. Lazy import and load; `local_files_only` unless `SATQUERY_VLM_ALLOW_DOWNLOAD=true`.
  - **Not exercised end-to-end yet:** that needs the [ml] extra and the weights (awaiting your go-ahead).
- Accuracy on the synthetic truth: water / built-up / vegetation / CVA / log-ratio masks all reach IoU > 0.9. Built-up gain hectares are within 3% of truth, and geodesic hectares on EPSG:4326 within 2%.
- 124 tests pass.

**Finding:** histogram matching (spec'd for temporal pairs) created about 1,200 false-change pixels over unchanged water. New built-up covering just 7% of the scene shifts the rank mapping for every other class. CVA now defaults to a robust (MAD-trimmed) linear gain/offset normalisation. Histogram matching stays available as `normalisation="histogram"`, and a test documents the failure mode.

### M4: Agentic controller & trace, `satquery/agent/` (2026-09-29)
- **Classifier.** `TaskClassifier` protocol + `RuleBasedClassifier` (keyword tables) → `TaskSpec` (task, targets, answer type, change direction, confidence, rationale).
  - Uses the validated input config as context: e.g. a temporal pair implies change, opt+SAR with a target implies fusion.
  - `TaskSpec` is a plain pydantic model, so its JSON schema can serve as an LLM function-calling schema later.
- **Planner.** Recipes (`mapping`, `fusion` / `single_source`, `class_change`, `generic_change`, `vlm`) are capability chains. `PlanBuilder` asks the registry for the best tool per capability, given the actual layers.
  - A higher-priority tool registered later is used automatically (tested).
  - If nothing fits, it explains why: `BAND_ROLE_MISSING` (e.g. built-up without SWIR), `MODALITY_MISMATCH`, `TOOL_UNAVAILABLE`, `NO_TOOL`.
  - Fallbacks, with warnings: fusion → single source; class change on SAR → generic log-ratio change.
- **Executor.** Runs steps in order and records effective params (defaults filled in). A crashing or invalid step fails alone; dependents are skipped.
- **Composer.** Grounded answers with hectares, share, regions and dates. Yes/no follows the asked direction.
  - Confidence heuristic is documented in `composer.py`: mask-step confidence, × cross-check IoU for class change (× 0.8 without it), fusion agreement, 0 for mocks, classical capped at 0.95.
  - Evidence lists masks per layer.
- **Trace.** `ExecutionTrace` schema v1.0: inputs (sha256, modality provenance, band sources, alignment, grid), task, plan with rationale, steps (effective params, runtime, output summary, confidence, mock flag, errors), answer, artifacts, warnings/errors, registry digest.
  - A trace is emitted for rejected requests too.
- **`analyze()`** never raises for user errors. Status is `ok` / `partial` / `rejected` / `error`. It writes quicklooks, RGBA mask layers, overlays, GeoTIFF masks, WGS84 GeoJSON and `trace.json`.
- 181 tests pass: 30 classifier queries, planner recipes and errors, executor isolation, end-to-end answers and trace-shape stability.

### M5: API & minimal UI (2026-09-29)
- **FastAPI** (`satquery.api.app:create_app`):
  - `POST /analyze`: multipart; 1–2 `files`, `query`, per-file `modality_N` / `date_N` / `band_map_N` (JSON, 1-based), `benchmark`.
    - Returns `{request_id, status, answer, confidence, error, artifacts[{kind,label,url,layer}], trace_url, trace}`.
    - HTTP 200 for ok/partial, 422 for rejected (the trace is still included), 413 for oversized uploads, 500 for internal errors.
  - `GET /tools` (registry export) and `GET /health` (includes VLM availability).
  - `GET /requests/{id}/files/{name}`.
- **Storage:** `data_dir/requests/<uuid>/{inputs,outputs}`, with sanitised filenames, uuid-only ids, a path-traversal-safe file route and a streaming upload cap (`max_upload_mb`).
- **UI** (`ui/`, Vite 8 + React 19 + TypeScript):
  - Upload form: 1–2 files, question, example chips, per-file overrides, benchmark toggle.
  - Answer card: confidence meter + basis, MOCK badge, structured errors with hints, warnings.
  - Evidence viewer: quicklook with toggleable mask layers and an opacity slider, side by side for pairs; GeoTIFF/GeoJSON downloads.
  - Trace viewer: steps table, planner rationale, full JSON with download.
- **just recipes:**
  - `just serve` builds the UI, and FastAPI serves it at `/` in a single process (works on Windows).
  - `just dev-api` + `just dev-ui` give hot reload (Vite proxies to the API).
  - `just demo-data` writes demo GeoTIFFs to `data/demo`. `just lint` also type-checks the UI.
- **Tests:** e2e API tests for SINGLE_OPT, SINGLE_SAR, PAIR_OPT_SAR and PAIR_TEMPORAL (answers checked against the truth, every artifact URL fetched). Also covered: 422 cases (change with one image, PNG without the flag, three files, bad overrides), band-map override, 413 upload limit, and path traversal.
  - 195 tests pass.
- **Manual check:** `just serve` served the built UI, and a real `curl` multipart `/analyze` on the demo temporal pair returned the built-up answer and working artifact URLs.
  - Not browser-tested yet: the Chrome extension wasn't connected in this session.

## Next
- M6: eval harness (`satquery/eval/`): dataset protocol + loaders, metrics, `satquery-eval` CLI; confirm RSVQA-LR download URLs before writing download code; README with architecture + "add a tool".

## Open questions / decisions log
- Starlette warns that `httpx` for its TestClient is deprecated (suggests `httpx2`). Harmless for now; revisit once the replacement is verified.
- Whole rasters are read into memory in Phase 1. Tiling utilities exist, but tools use them only above `tile_size`.
- Lee filter only. Refined Lee is a TODO (Phase 2).
- Built-up needs SWIR (NDBI). 4-band inputs will get a structured `BAND_ROLE_MISSING` error (decided).
- Old pip `venv/` in the repo root is unused (uv uses `.venv/`); it's git-ignored and safe to delete.
- M6: confirm RSVQA-LR download URLs and the CDVQA image hosting before writing download code.
