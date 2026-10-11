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

### M6: Evaluation harness, `satquery/eval/` + README (2026-09-29)
- **Sources** (`eval/sources.py`), each checked against its primary source:
  - RSVQA-LR: Zenodo record 6344334 (concept DOI 10.5281/zenodo.6344333, CC-BY-4.0, 150.5 MB).
  - VRSBench: HF `xiang709/VRSBench` (CC-BY-4.0).
  - CDVQA: github.com/YZHJessica/CDVQA (annotations only).
  - BigEarthNet.txt: HF `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (CDLA-Permissive-1.0).
- **Loaders**: `BenchmarkDataset` protocol with a deterministic `subset(n, seed)`. Missing files raise `DatasetNotFoundError` naming the source and the expected files.
  - `rsvqa-lr` and `cdvqa` share one RSVQA-style schema, verified against the authors' `VQALoader.py` and the CDVQA JSONs.
  - `vrsbench-{vqa,caption,grounding}`: JSON lists; boxes parsed from `{<x1><y1><x2><y2>}` on a 0–100 scale; images found by name anywhere under the root.
  - `bigearthnet-txt`: parquet columns from the dataset's own datamodule; S2 bands stacked from a local BigEarthNet v2 copy.
- **Metrics**:
  - VQA: overall and average accuracy, per type, with answer normalisation.
  - Captioning: BLEU-4 / METEOR / CIDEr via pycocoevalcap, simple tokeniser; METEOR uses Java, which is present.
  - Grounding: Acc@IoU ≥ 0.5 and mean IoU.
  - Change masks: precision / recall / F1 / IoU, micro and per-image.
- **Runner + CLI**: `satquery-eval --benchmark B --tool T|pipeline --subset N [--seed --split --data-root --out]`.
  - Evaluates one registry tool (checked by *capability*, not just task) or the full pipeline.
  - Writes `results/<bench>_<tool>_<ts>.{json,md}` with tool version, mock flag, registry digest, git commit, subset and seed, and per-sample records.
- **README**: setup, settings, mermaid architecture, API, "adding a tool" walkthrough, evaluation.
- 217 tests pass (loaders and runner run on tiny synthetic copies of each benchmark's file layout). `just lint` (ruff + UI type-check) is clean.

### Follow-up: real benchmarks, downloads and zero-shot VLM (2026-09-29)
- **Downloads.** `satquery-eval --download rsvqa-lr | cdvqa` (also `just download <name>`) streams to `.part` and resumes safely.
  - RSVQA-LR: files are checked against the MD5s read from the Zenodo API. 772 images plus the train/val/test JSONs.
  - CDVQA: annotations from GitHub. Images are the SECOND train set from the SECOND authors' page (captain-whu.github.io/SCD), a 2.41 GB RAR that is size-checked and extracted with the libarchive `tar` (Git's GNU tar can't read RAR).
    - I read the listing of the larger Google Drive zip remotely first: it only bundles this RAR and a test zip, so it isn't needed.
    - 2,968 pairs; every image referenced by test and test2 resolves.
- **Planner fallback** (found by running RSVQA-LR). On RGB-only images the classical tools can't map water / built-up / vegetation (no NIR/SWIR), and the pipeline rejected 87% of questions.
  - Now yes/no, count and free-text questions fall back to the vision-language tool, with a warning that says why.
  - Area questions are still rejected with the actionable `band_role_missing` error (only a mask gives hectares).
  - Errors on the 500-question subset went from 437 to 8.
- **`[ml]` extra** now includes `torchvision` (Qwen3-VL's processor imports its video processor). Installed: torch 2.14 (CPU), transformers 5.17. Weights for `Qwen/Qwen3-VL-2B-Instruct` are in the HF cache (4.0 GB).
  - `just test-ml` (pytest `--run-ml`) passes with real weights: VQA, caption, two-image change.
  - VQA prompts ask for one-word / number answers. Prompts were not tuned beyond a 10-question sanity probe (avoids test-set leakage).
- **Eval reports** are split into a summary `.json` (committed), `.records.jsonl.gz` per-sample records (3.6 MB → 186 KB for the full RSVQA-LR test) and `.md`.
- **Baselines** are in `results/README.md`.
  - RSVQA-LR full test, mock: OA 39.9 / AA 30.9. On the 500-question subset (seed 0), mock and pipeline both score 42.8 / 32.0.
  - CDVQA 1,000-question subset: mock and pipeline both score 29.2 / 14.4. Explanation in the open questions below.
  - Qwen3-VL-2B zero-shot on RSVQA-LR (500, seed 0): OA 47.8 / AA 33.8, vs 42.8 / 32.0 for the mock prior. Comparison 71.5 (mock 30.1); presence 63.9 (mock 79.5); count and rural/urban 0 (see open questions). 54 min on CPU.
  - Qwen3-VL-2B zero-shot on CDVQA (1,000, seed 0, GPU): OA 42.5 / AA 24.1, vs 29.2 / 14.4. On yes/no change questions 67.9% (always-"yes" 47.1%). Categorical and ratio types 0 (prompt allows only yes/no or a number). 7.8 min on an RTX 4060 Laptop GPU.
- **GPU support.** Torch now comes from two mutually exclusive extras, `cpu` (PyTorch CPU index; PyPI on macOS) and `cuda` (PyTorch CUDA 12.6 index, driver ≥ 560), both locked in `uv.lock`.
  - `just install-ml` installs CPU; `just install-gpu` installs CUDA.
  - Before this, PyPI's torch was CPU-only on Windows but bundled CUDA 13 on Linux, so "CPU install" meant different things per OS.
  - Qwen loads on `cuda:0` in bf16 (4.26 GB VRAM). It runs at 0.46 s per two-image question, vs ~6.4 s per single-image question on CPU.

### Docs: plain-language guide (2026-09-29)
- `PROJECT_GUIDE.md`: non-technical walkthrough (glossary with full forms, request journey, tool list, folder tour, how to run on Windows, limits).

# Phase 2.1: Qwen3.5-4B LoRA for BigEarthNet.txt VQA (`src/benvqa/`)
Plan, verified facts and deviations: `docs/phase2_vqa_plan.md`.

### M0: Recon & plan (2026-10-11)
- Verified the dataset, Zenodo, model and template facts against primary sources; written up in the plan.
- New facts: the model needs `mm_token_type_ids`; MTP weights are not even loaded by transformers; the `unsloth/` repo ships a different tokenizer; S1 tiles look like dB already.

## Definition of done: status
- [x] `just test` passes on CPU-only (no torch); `just serve` starts API + UI.
- [x] All four input configurations work end-to-end on synthetic data with a clean trace (e2e API tests).
- [x] Water / built-up / change queries return real classical results with hectares and overlays; VLM tasks return labelled mock outputs through the same pipeline.
- [x] Eval CLI runs end-to-end on real benchmark subsets (RSVQA-LR, CDVQA) with mock, pipeline and zero-shot Qwen tools.
- [x] README with setup, mermaid architecture and how to add a tool.

## Next
- Re-run the RSVQA-LR zero-shot baseline on the GPU, on a larger subset or the full split (~80 min at GPU speed, estimated).
- A per-question-type answer format for the VLM prompt (yes/no, number, or a word such as rural/urban or a CDVQA class name). Tune on the **val** splits, never on test.
- VRSBench download command (HF `xiang709/VRSBench`) for caption / grounding baselines.
- Phase 2 candidates:
  - an LLM function-calling classifier behind `TaskClassifier`;
  - learned water/built-up/change models registered at higher priority;
  - a tiled executor for large rasters;
  - refined Lee;
  - multi-target queries.

## Open questions / decisions log
- Fixed (2026-10-11): `src/satquery/eval/datasets/` was untracked because the `.gitignore` rule `datasets/` matched it; the rule is now `/datasets/` and the package is committed.
- **CDVQA licence:** the SECOND page states no licence or terms of use for the images. Check before redistributing images or publishing derived results.
- **CDVQA pipeline = always "yes".**
  - Otsu thresholding always splits the CVA histogram into two classes, so some change is always detected.
  - Class-specific questions ("did the trees change?") can't be mapped on RGB.
  - CDVQA's categorical answers (`NVG_surface`, `0_to_10`) are not produced.
  - Needs a learned change model and a benchmark answer adapter (Phase 2).
  - A bimodality/no-change test for CVA is a cheap improvement to consider.
- **Qwen3-VL-2B zero-shot answers only yes/no on RSVQA-LR.**
  - It answers "yes"/"no" even to "What is the amount of X?", so count accuracy is 0.
  - Rural/urban is also 0: the prompt restricts replies to yes/no or a number, which excludes "rural"/"urban". That's a prompt artifact.
  - Left as is so the baseline isn't tuned on test; fix it via the val split.
- The Qwen tool runs at ~6 s per question on CPU and ~0.5 s on the RTX 4060 (`just install-gpu`).
- After `just install-gpu`, use `uv run --no-sync ...` (or pass the same extras). A plain `uv sync` / `just install-ml` switches torch back to the CPU build.
- BigEarthNet.txt: per-band file naming inside BigEarthNet v2 patch folders is assumed (`<patch>_B02.tif` ...); verify on a real copy. Grounding rows are skipped until their box format is verified.
- UI not browser-tested in-session (Chrome extension not connected). API verified with real requests.
- Starlette warns that `httpx` for its TestClient is deprecated (suggests `httpx2`). Harmless for now; revisit once the replacement is verified.
- Whole rasters are read into memory. Tiling/stitching utilities exist and are tested, but tools don't use them yet (fine for typical scene sizes; needed for very large rasters).
- Lee filter only. Refined Lee is a TODO.
- Decided: built-up needs SWIR (NDBI), so 4-band inputs get a structured `band_role_missing` error.
- Decided: CVA uses robust linear radiometric normalisation by default; histogram matching invents change (see M3).
- Old pip `venv/` in the repo root is unused (uv uses `.venv/`); it's git-ignored and safe to delete.
