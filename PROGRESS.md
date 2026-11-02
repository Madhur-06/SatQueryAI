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

### M1: Dataset analysis (2026-10-11)
- `scripts/analyze_dataset.py` → `reports/dataset_analysis.{md,json}` (74 s on CPU, parquet only).
- Every row count in the brief reproduces exactly. No image is shared between any two splits (S2 or S1 names).
- MCQ parser (`benvqa.data.mcq`) round-trips 100% of 3,259,184 MCQs. All have 4 options; 39% have commas inside options.
- Differences from the brief:
  - test has 115,753 images (116,835 = test + bench);
  - adjacency `yes`/MCQ exist for 79% of train images, while the adjacency `no` binary exists for all. Group picks are therefore absolute probabilities, and an unavailable pick skips the group;
  - bench has a different per-image structure (many questions per category).
- Projection: about 5.75 QAs per conversation, 590 tokens vs 322 single-turn → 3.1× answers per token.

### M2: Shard builder (2026-10-11): code + dry run done, full build pending (needs your HF account)
- `benvqa.data.build_shards`:
  - parquet → per-image selection: train 229,114 / val_small 3,000 / bench 1,082; test is never selected (asserted);
  - pass 1 streams S1 into a memmap;
  - pass 2 streams S2 into 32 mixing buckets → `<split>/shard-XXXXX.parquet`. Each shard is one Hub commit with its patch list and the updated `meta/shard_index.json`;
  - resume = union of the uploaded patch lists;
  - `verify()` checks the index totals against the parquet.
- Verified on the real archive:
  - S2 is uint16 with **no** BOA offset (B02 minima 70–90 DN, baseline N9999); 20 m bands are 60×60;
  - S1 is float32 **dB**;
  - layout is `BigEarthNet-S2/<tile>/<patch>/<patch>_<band>.tif`.
- **Zenodo throttles each connection to about 0.4 MB/s.** A single stream would need more than 3 days for 117.7 GB. New `ParallelRangeReader`: N Range connections, ordered chunks, per-chunk retries. 16 connections gave 5.4 MB/s here, and the Kaggle notebook uses 32. `ResumableHTTPStream` (Range-resume) remains for single-connection use.
- Dry run (first 2,000 members of each tar) passes end to end. Only 3 of those patches are selected (the early tiles are mostly test/validation).
- Dev mini build for local smoke tests, in `data/benvqa/dev_hub` (local only): 910 train + 507 validation images with real S1, ~1 GB pulled.
- Previews (`reports/previews/dev_train.jpg`) look sane; saturation is under 0.5% in every channel.
- Shards are about 130 KB per image at zstd 9 (level 6 gives 137, level 15 gives 129), so a 1,000-image shard is ~130 MB and the full build is **~30 GB**.
- 12 tests:
  - quantization, upsampling, selection;
  - end-to-end build from synthetic `tar.zst`;
  - crash → resume without duplicates;
  - flaky HTTP (single and parallel), range reader.
- `notebooks/kaggle_build_shards.ipynb` (generated by `scripts/make_notebooks.py`), `just kaggle-code` (code zip for Kaggle).
- Fixed: the `.gitignore` rule `data/` also hid `src/benvqa/data/`; it is now `/data/`.

### M3: Conversation builder (2026-10-11)
- `benvqa.data.conversation`:
  - leakage groups, one item per group;
  - absolute pick probabilities (an unavailable pick skips the group);
  - presence placed before any class-mentioning turn; `p_drop_presence` 0.3;
  - caption last with `p_caption` 0.2;
  - per-visit MCQ option shuffling;
  - modality dropout;
  - randomness is a pure function of `(seed, patch_id, visit)`; kinds cycle across visits so re-visits see new items.
- `benvqa.data.render`:
  - segment-by-segment tokens: every assistant turn opens with the exact non-thinking generation prefix;
  - labels = answer + `<|im_end|>`, with per-token turn index and per-turn weights;
  - truncation drops the caption first.
- `benvqa.data.vision`: numpy patchify (uint8 tiles); GPU-side normalisation reproduces the official processor exactly.
- Tests (17), all passing with the cached Qwen3.5-0.8B tokenizer/processor:
  - leakage rules over 50k conversations (incl. bench-like multi-item images);
  - statistics match the config (MCQ 0.5, presence kept 0.7, caption 0.2, adjacency yes 44%, ~5.75 QAs);
  - MCQ shuffle remaps answers correctly; determinism; visit cycling;
  - single-turn render == official template;
  - segment-wise == whole-string tokenization (300 conversations);
  - label mask; turn weights; single-token answers; truncation; 0.8B and 4B tokenizers agree;
  - patchify bit-exact.
- Fixed a Phase 1 test that assumed no earlier test imports transformers (it now hides the module and checks the tool doesn't import it).

### M4: Streaming loader (2026-10-11)
- `benvqa.config`: one pydantic schema for the YAML config (`configs/kaggle_t4x2.yaml`, `configs/smoke_08b.yaml`), `key=value` overrides, and a `data_hash` that resume must match.
- `benvqa.data.stream`:
  - G = world × workers streams, each a pure function of `(seed, stream_id)`: per-epoch shard permutation, groups of 2 interleaved shards, length-bucketed pools of 64;
  - the cursor counts micro-batches; resume skips whole groups/pools arithmetically and rotates worker→stream so DataLoader round-robin continues exactly;
  - a background `ShardFetcher` per stream (look-ahead, reference-counted deletion, budget);
  - train-only assertions (config, shard paths, row `split`).
- `benvqa.data.collate`: right-padded batch, `mm_token_type_ids`, uint8 tiles, label positions only, turn ids/weights/categories.
- Tests (9):
  - plan partitions and `locate` arithmetic;
  - batch invariants;
  - exact resume (next 100 micro-batches identical after 37 and 64 consumed, across epochs);
  - 2-worker DataLoader order == in-process order;
  - disjoint ranks;
  - 40% injected download failures → identical stream;
  - consumed shards deleted;
  - non-train data rejected.
- Throughput on real dev shards (`reports/throughput_loader.json`): **472 conversations/s** with 2 workers (2,785 QAs/s), 0.95% padding, 5.89 QAs and 576 tokens per conversation. Per-conversation CPU cost is 4.6 ms, mostly composites + resize + patchify. That is ~300× what one rank needs.

### M5: Trainer + Kaggle packaging (2026-10-11): ready for the Kaggle M5b smoke run
- `benvqa.train` (`torchrun -m benvqa.train`):
  - DDP (nccl/gloo), fp16 autocast + GradScaler, LoRA weights in fp32;
  - global turn-weight normaliser (one all-reduce per step);
  - label-position-only LM head (`model/loss.py`);
  - WSD schedule (`schedule.py`, `--decay-now`);
  - rank-0 control broadcast (checkpoint / eval / stop);
  - NaN guard (skip; after 3 in a row reload + halve lr); session guard (`SESSION_START_EPOCH`, SIGTERM);
  - JSONL log (+ W&B); in-training score-based val eval; `best/` adapter.
- `benvqa.checkpoint`:
  - atomic writes (tmp + fsync + rename) and atomic `LATEST`; keeps 2 locally and 3 on the Hub;
  - async Hub mirror: busy uploader → stays local, and `finalize` uploads the newest;
  - in-flight folders are never pruned; resumes from the mirror on a fresh machine.
- `benvqa.model.load`:
  - `hf` backend (transformers + PEFT, verified);
  - `unsloth` backend written to the documented API (unverified; Windows);
  - on CPU, transformers' bindings to the GPU-only fla kernels are swapped back to torch.
- `benvqa.evaluation` + `benvqa.eval_bench`:
  - single-turn score-based VQA with softmax confidence; paper-style per-category table;
  - greedy captions (BLEU-4/METEOR/ROUGE-L/CIDEr; ROUGE-L added to `satquery.eval.metrics.caption`);
  - reliability diagram SVG (validated palette, light/dark);
  - runs a–e → `reports/bench_results.md`.
- Tests (CPU, tiny random Qwen3.5 with the real architecture): 25 in `test_train.py`, including:
  - label-position loss == full-logits loss; right-padding invariance; WSD shape;
  - checkpoint rotation + mirror resume; busy uploader;
  - **hard kill (os._exit) → resume gives identical per-step losses, lr, cursors, counters and final adapter**;
  - resume from the mirror with local checkpoints deleted; **2-process DDP**; eval_bench a + c.
- Real Qwen3.5-0.8B on RTX 4060 (`reports/smoke_qwen35-08b_checks.json`):
  - label-position loss == reference (diff 0.0); right padding 1e-4 relative;
  - fp16 vs fp32 loss diff 0.03, all finite;
  - 20-step run hard-killed at step 12, resumed from step 10: step 11 loss identical (0.6710);
  - loss 0.91 → 0.40, val mean OA 41.7 → 63.8 (64 questions).
- Fast kernels (`reports/kernel_check.json`): fla/Triton is 1.84× faster fwd+bwd than the torch fallback (loss diff 0.003, grad cosine 0.99991). Tested locally via triton-windows + fla-core, installed into the venv outside the lockfile.
- Bugs found and fixed:
  - `from_pretrained` returns eval mode, which silently disabled gradient checkpointing (7.8 vs 3.6 GB peak);
  - Triton recompiles per sequence length (new `train.pad_to_multiple: 64`; 1.35 s → 0.65 s per micro-batch);
  - `LocalMirror.list` prefix semantics; Windows fsync; DDP test harness pipes.
- Kaggle: `requirements-kaggle.txt` (pinned), `notebooks/kaggle_train.ipynb` (MODE smoke/train/eval), `scripts/kaggle_smoke.py` (M5b: env, micro-batch sweep, Hub round trip, recommended `planned_images`, tested locally), `docs/kaggle_runbook.md`.

### M5b: Kaggle smoke on 2×T4 (2026-10-24): sweep passed, round trip found a resume bug (fixed)
- Env: Python 3.13, torch 2.11+cu128. `fla` imports and runs on Turing; `causal_conv1d` absent (torch fallback, ~2%).
- Sweep (32 conversations/step, 12 steps each, training read from the Hub shards): every setting exited 0, no OOM, finite loss (0.74–0.76 → 0.56–0.65), `data_wait_frac` ≤ 0.004.
  - mb 1 / 2 / 4: 0.704 / 0.738 / 0.753 conversations/s, peak 10.28 / 11.17 / 12.89 GB, ~43–45 s per step. GPU-bound: micro-batch barely matters.
  - The first step of a fresh session takes ~330 s (Triton compilation).
  - 45 h at 0.75 conv/s → `planned_images` ≈ 122k (about half the 229k selected train images).
  - The script recommends mb 4, but mb 2 is only 2% slower with 1.7 GB more headroom. The sweep only saw ~384 conversations of ~576 tokens, while `max_len` is 1024; a rough extrapolation puts a long mb 4 batch near the T4's 15 GB. Prefer `train.micro_batch=2 train.grad_accum=8`.
- **Bug (fixed): resume from a `hub:` dataset always failed with `data_hash mismatch`.** The first run resolves `data.revision` (Hub head) and saves it in the hash; the resumed config has no revision, and `_check_compatible` ran before the saved revision was restored. Tests used `local:` mirrors, where the revision stays `None`, so they never saw it. Every session after the first would have crashed at startup.
  - Fix: `Trainer` restores the saved revision (when none is given) before the check; an explicit different revision still fails the check.
  - Test: `test_resume_restores_data_revision` pins a revision on the first run (stand-in for the Hub lookup), resumes without it, and checks that a different explicit revision is rejected. It failed with the Kaggle error before the fix.
  - The first round-trip run also died after 28 s, before it could have saved anything, so `benvqa-smoke` most likely held a checkpoint from an earlier attempt. Delete that repo (or its `ckpt/`) before re-running the round trip.
- **Re-run with the fix (2026-10-29): M5b passed.** `reports/m5b_smoke.json`.
  - Hub round trip: both runs exit 0, `resumed_from_hub: true` (local checkpoints deleted, resumed from the Hub).
  - Sweep reproduces the first run: mb 1 / 2 / 4 at 0.717 / 0.747 / 0.759 conversations/s, the same peak memory, and bit-identical first-step losses (deterministic data order and seeding across sessions); last-step losses differ by ~1e-3 (GPU kernel nondeterminism).
  - Training settings: mb 2 → 0.747 conv/s × 45 h ≈ 121k conversations, i.e. `train.micro_batch=2 train.grad_accum=8 train.planned_images=121014`.

### M6: Zero-shot bench baselines on 2×T4 (2026-10-29)
- `MODE=eval`, runs a,b on the full bench split (12,477 VQA questions, ~1.5 h, commit `0c01eb2`). Reports were uploaded to `reports/` in the `<HF_USER>/qwen35-4b-benvqa-lora` Hub repo. The M8 eval downloads them again, so its `bench_results.md` lists a–e together.

  | Model | Binary OA | MCQ OA |
  |---|---|---|
  | (a) Qwen3.5-4B zero-shot, RGB | **59.33** | **37.35** |
  | (b) Qwen3.5-4B zero-shot, RGB + NIR-SWIR + SAR | 57.98 | 35.80 |
  | Qwen3-VL-8B zero-shot (paper, best open model) | 61.96 | 37.55 |
  | GPT-5.2 zero-shot (paper) | 60.39 | 34.93 |
  | RS-InternVL fine-tuned (paper) | 73.29 | 51.49 |
  | Trivial (always "yes" / most common letter) | 48.0 | 25.7 |

- Yes-bias: the model answers "yes" to 71% of binary questions, against 48% in the gold answers. It gets 82% of "yes" questions right but only 39% of "no" questions.
- The extra composites hurt the untrained model (b < a, and "yes" rises to 75%). The key fine-tuned comparison is (c) against (a).
- MCQ count (27.1) and S (25.1) are at chance (~25).
- Overconfident: ECE ≈ 0.12, and answers given with >95% confidence are right 76% of the time. Calibrate on val before showing confidence in the app (Phase 3).

### M7: LoRA training on Kaggle 2×T4 (2026-10-30 → 2026-11-01): done, 5 sessions, 3,783 steps
- Settings for every session: `MODE="train"`, `DECAY_NOW=False`, `EXTRA = ["train.micro_batch=2", "train.grad_accum=8", "train.planned_images=121014"]` (3,782 steps × 32 conversations; WSD decay over the last 567 steps, starting at step 3,215).
- Every session resumed from the Hub checkpoint of the one before. Every session also had 0 skipped steps, no data wait and peak memory ≤ 11.57 GB, and ended on the session time guard after ~11.1 h, followed by val, checkpoint and upload. Grad-norm was NaN only in GradScaler overflow windows early in session 1 (steps 10, 60, 240); the loss scale went 512 → 128 → 256.

  | Session | Dates | Steps | conv/s | Loss (end) | lr (end) |
  |---|---|---|---|---|---|
  | 1 | 10-30 | 0 → 911 | 0.73 | ~0.18 | 1e-4 |
  | 2 | 10-31 | 911 → 1,753 | 0.68 | ~0.16 | 1e-4 |
  | 3 | 10-31 → 11-01 | 1,753 → 2,624 | 0.71 | ~0.14 | 1e-4 |
  | 4 | 11-01 | 2,624 → 3,477 | 0.69 | ~0.13 | 5.4e-5 (decaying) |
  | 5 | 11-01 | 3,477 → 3,783 (3.9 h) | 0.71 | ~0.115 | 0 |

- Throughput varies by ~7% between Kaggle machines, so a session covers 840–910 steps instead of the planned ~910. Session 5 stopped with `planned images reached` after 121,044 conversations (695k QAs, 66M tokens), about half a pass over the 229k selected train images. It ran one step past the planned 3,782, at lr 0 (harmless).
- In-training val (2,000 val_small questions; in-distribution, not comparable to the bench numbers):

  | Step | Binary | MCQ | Mean |
  |---|---|---|---|
  | 911 | 83.23 | 81.61 | 82.42 |
  | 1000 | 81.89 | 82.13 | 82.01 |
  | 1753 | 83.98 | 84.61 | 84.30 |
  | 2000 | 85.33 | 84.31 | 84.82 |
  | 2624 | 84.73 | 85.29 | 85.01 |
  | 3000 | 84.13 | 86.56 | 85.35 |
  | 3477 | 83.68 | 86.41 | 85.05 |
  | **3783** | **84.88** | **87.91** | **86.40** (final = `best/`) |

  - Train accuracy tracks val, so there is no sign of overfitting.
  - MCQ keeps improving (area 68 → 75–78, season 72 → 96, climate zone 92 → 96). MCQ adjacency (~75) is the weakest category.
  - The decay gave the expected final gain: +1.35 mean OA from step 3,477 to 3,783, and loss fell from ~0.133 to ~0.116.
  - The binary dip (85.33 → 83.68 over steps 2000–3477) recovered to 84.88 at the end, so it was mostly noise. MCQ relative position rose to 85.5 and adjacency to 77.8; MCQ area (77.8) and count (80.8) are now the weakest categories.
- Kaggle gotcha: the notebook viewer shows the output of the last *successful* version. Pick the newest version in the version list before downloading, or you get an older session's logs.

### M8: Fine-tuned bench eval on 2×T4 (2026-11-02)
- `MODE=eval`, runs c,d,e + 200 captions per run, on the full bench split (12,477 VQA questions). The adapter is `ckpt/best` = step 3,783 (val mean 86.40). Wall time ~5.2 h: VQA took about as long as in M6, plus ~1 h (57–65 min) of unbatched greedy captions per run.
- `reports/bench_results.md` holds the full a–e table; also `bench_{a..e}.json` and `reliability_{a..e}.svg`. The per-question records (`*.records.jsonl.gz`, ~0.5 MB each) stay on the Hub (`reports/`).

  | Model | Binary OA | MCQ OA | BLEU-4 | ECE (bin / MCQ) |
  |---|---|---|---|---|
  | (a) zero-shot, RGB | 59.33 | 37.35 | - | 0.12 / 0.12 |
  | RS-InternVL fine-tuned (paper) | 73.29 | 51.49 | 34.04 | - |
  | **(c) LoRA, RGB + NIR-SWIR + SAR** | **77.23** | **78.97** | 0.42 | 0.06 / 0.06 |
  | (d) LoRA, RGB only | 76.92 | 78.63 | 0.40 | 0.06 / 0.07 |
  | (e) LoRA, SAR only | 67.00 | 61.57 | 0.14 | 0.11 / 0.10 |

- Fine-tuning gave +17.9 binary and +41.6 MCQ over zero-shot.
- **Yes-bias gone:** the model now answers "yes" 46.9% of the time (gold 48.0%). Accuracy on "no" questions went from 39% to 79%.
- **Calibration halved:** ECE 0.12 → 0.06, without any explicit calibration.
- **Extra composites add nothing measurable:** c − d is +0.31 / +0.34, and McNemar χ² is 2.5 / 2.9, so p ≈ 0.1. The adapter works almost entirely from RGB, and training used modality dropout.
- **SAR alone carries real signal:** it beats zero-shot RGB by +7.7 / +24.2, and area (80) and count (78) are close to RGB. Country (58.7), season (65.3) and binary presence (58.0) need optical colour.
- Per category (c):
  - Strongest: MCQ country 94.9, season 97.0, climate zone 96.7. These are tile- and date-level cues and the easiest to learn.
  - MCQ excluding those three: about 72.
  - Weakest: MCQ count 60.2, MCQ adjacency 66.4, binary adjacency 72.6.
- Bench is 7–9 points below val (84.88 / 87.91). That's expected: the bench is manually verified and asks many questions per image. M1 verified that no image is shared between splits, and test/bench images are never selected for training.
- Captions copy the dataset's caption template almost word for word (~79 words against ~83 in the reference). The errors are in the facts: area values and sometimes the season. That template matching is why n-gram scores are high. Our BLEU-4 (0.42 on a 0–1 scale) uses a simple lower-case tokenizer and only 200 captions, so it is **not** comparable to the paper's 34.04 (PTB tokenizer, 0–100 scale).
- **Caveat before claiming "beats RS-InternVL":** our VQA is score-based. The model picks among the offered candidates by logits, so it can never give an invalid answer. The paper most likely generated free text. Zero-shot under our protocol matches the paper's zero-shot models (MCQ 37.35 vs 37.55), so the protocol isn't obviously inflating scores. Still, a +27.5 MCQ claim needs a generative check on a bench subset first.

## Next (Phase 2.1)
1. Generative sanity eval: ~1,000 bench questions, free generation, answers parsed and exact-matched. Compare with the score-based accuracy on the same questions before quoting the comparison with the paper.
2. Phase 3: register the adapter as a `benvqa_lora` tool, with confidence calibrated on val.
3. Speed up `generate_captions`: batch generation and a lower `--max-new-tokens`, since captions are ~80 words.
4. Optionally try `BACKEND=unsloth` in a second smoke run.

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
- **`--decay-now` (`DECAY_NOW=True`) wastes the rest of the session.** It decays over min(15% of planned steps, steps that fit in the session) and then trains at lr 0 (`min_lr_ratio: 0`) until the time guard. With 567 decay steps against ~880 that fit, that's ~4 h at lr 0. Not used in M7. Fix: decay over the whole session, or stop once lr reaches 0.
- Old pip `venv/` in the repo root is unused (uv uses `.venv/`); it's git-ignored and safe to delete.
