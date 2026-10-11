# Phase 2.1 plan: Qwen3.5-4B LoRA for single-image VQA on BigEarthNet.txt

Brief: fine-tune a 16-bit LoRA adapter on Qwen3.5-4B for binary + MCQ VQA, with captioning
as a secondary task. Budget is about 30 GPU-hours a week on Kaggle 2×T4. The aim is the
most supervised answers per GPU-second, with training that survives crashes and session
limits. Targets (bench split): binary 73.29 / MCQ 51.49 (RS-InternVL, fine-tuned).

This document is the working plan. Section "Deviations" records every place where
practice forced a change from the brief.

## Verified on 2026-10-11 (M0)

| Fact | Source | Status |
|---|---|---|
| `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`: one `BigEarthNet.txt.parquet`, 466,819,745 bytes, CDLA-Permissive-1.0 | HF API (`repo_info`, rev `72d865f`) | matches brief |
| Zenodo 10891137: `BigEarthNet-S1.tar.zst` 54.44 GB, `BigEarthNet-S2.tar.zst` 63.25 GB, `metadata.parquet` 3.6 MB, md5s available | Zenodo REST API | matches brief (sizes in GB, not GiB) |
| `Qwen/Qwen3.5-4B` (Apache-2.0): 32 layers, `full_attention_interval` 4, hidden 2560, vocab 248,320, tied embeddings, `mtp_num_hidden_layers` 1; vision depth 24, patch 16, merge 2; checkpoint 9.32 GB (bf16) | `config.json` | matches brief |
| Processor: `Qwen3VLProcessor`, `Qwen2VLImageProcessorFast`, `shortest_edge` 65,536, mean/std 0.5 | `preprocessor_config.json` | matches brief |
| transformers 5.17 has `qwen3_5` natively. The MTP weights are dropped at load (`_keys_to_ignore_on_load_unexpected = ["^mtp.*"]`), so there is nothing to freeze | `modeling_qwen3_5.py` | better than expected |
| The model needs `mm_token_type_ids` for M-RoPE whenever images are passed (it raises otherwise) | `modeling_qwen3_5.py` `compute_3d_position_ids` | **new**: the collator must emit it |
| GatedDeltaNet and its causal conv are causal and zero padded positions (`apply_mask_to_padding_states`), so **right** padding cannot affect real positions. Kernels come from `fla` / `causal_conv1d` via the hub-kernel fallback, with a pure-torch fallback otherwise | `modeling_qwen3_5.py` | still to verify numerically (M5) |
| Chat template: historical assistant turns render **without** a think block. Only turns after the last user message get `<think>\n\n</think>\n\n`. Non-thinking generation prompt = `<\|im_start\|>assistant\n<think>\n\n</think>\n\n`. 4B defaults to thinking and 0.8B to non-thinking; both are identical with `enable_thinking=False` passed explicitly | `chat_template.jinja` (both) | confirms §6.3: build segment by segment |
| Image items render as `<\|vision_start\|><\|image_pad\|><\|vision_end\|>`, concatenated to adjacent text with no separator; user content is `trim`med | template | used for the render-equivalence test |
| `unsloth/Qwen3.5-4B` ships a **different** `tokenizer.json` (20.0 MB vs 12.8 MB) and chat template | HF file listing | **risk**: always take tokenizer/processor from `Qwen/Qwen3.5-4B`, and test that ids match |
| BigEarthNet v2 S1 means are VV −12.6 / VH −19.4, so the S1 GeoTIFFs are already in **dB**. S2 B02 mean 438 suggests **no** +1000 BOA offset | official `ben_txt_datamodule.py` band stats | re-verify on raw tiles in the M2 dry run |

## What we reuse from Phase 1

| Existing | Use in Phase 2 |
|---|---|
| `satquery.eval.sources.SOURCES["bigearthnet-txt"]` | Single source of truth for dataset provenance. Zenodo S1/S2 entries are added there |
| `satquery.eval.datasets.bigearthnet_txt` (`task_for`, parquet schema, `datasets/bigearthnet-txt/` root convention) | Same on-disk location for the parquet; `task_for` is reused to classify rows |
| `satquery.eval.metrics.vqa.normalize_answer` / `vqa_accuracy` | Per-category accuracy in `eval_bench` (same normalisation as every other benchmark) |
| `satquery.eval.metrics.caption.caption_scores` | BLEU-4 / METEOR / CIDEr for bench captions. **ROUGE-L is added there** (the paper reports it) |
| `satquery.eval.runner` conventions (git commit, subset+seed, summary JSON + gz records + md) | `eval_bench` writes the same three-file layout under `reports/` |
| `satquery.tools.ml.qwen_vl` | Pattern for Phase 3: the trained adapter becomes a registry tool (`benvqa_lora`) with the score-based confidence. Not part of 2.1 |
| pytest markers (`ml`, `--run-ml`), uv extras, ruff config, `just` recipes | New `benvqa` tests follow the same rules: CPU-only, synthetic data, skip when an optional dep or cached tokenizer is missing |

The Phase 1 BigEarthNet loader reads per-band GeoTIFFs from a local v2 copy, one row at a
time. Training needs the opposite (no local copy, per-image grouping, uint8 cubes), so
the image path is new code: `build_shards` and `stream`. The bench split is evaluated
from the shards, not through the Phase 1 loader.

## Design decisions beyond the brief

1. **Package location `src/benvqa/`** (not top-level `benvqa/`). This matches the repo's
   src layout and hatch config. Kaggle runs it with `PYTHONPATH=src` (no install step for
   the satquery stack). `python -m benvqa.train` works unchanged.
2. **HTTP streaming survives drops without restarting the pass.** `ResumableHTTPStream`
   is a file-like object. When the connection breaks it reconnects with
   `Range: bytes=<consumed>-` and keeps feeding the *same* zstd decompressor and tar
   reader, which never notice. Only a process restart falls back to "re-stream and skip
   done patches" (journal). `httpx` is used instead of `requests`, because it is already a
   dependency of `huggingface_hub` 1.x.
3. **Image preprocessing in numpy.** `benvqa.data.vision.patchify` reproduces
   `Qwen2VLImageProcessorFast`: rescale, normalise, temporal duplication, and the
   9-axis patch transpose. A test checks it bit-exact against the official processor.
   Workers therefore need no torchvision, and one 256² image costs about 0.2 ms.
4. **The data cursor counts micro-batches, not samples.** Length bucketing reorders
   samples inside a pool, so "N samples trained" is not a prefix of the stream. Instead
   the stream of *micro-batches* is a pure function of `(seed, stream_id)`:
   - pools (64 conversations) never cross a shard-pair boundary;
   - the micro-batches per pool are `ceil(pool/B)`, known from `shard_index.json`.

   Resume skips whole shard pairs and whole pools arithmetically (no download). Only the
   current pool is rebuilt (text only, about 64 conversations). Worker→stream assignment
   is rotated by `consumed % num_workers`, so the DataLoader's round-robin resumes on
   exactly the stream that was next.
5. **Global turn-weight normalisation.** At the start of each optimizer step the trainer
   pulls all `grad_accum` micro-batches (CPU tensors, already prefetched) and sums their
   turn weights. One all-reduce gives `W_global`. Each micro-batch loss is
   `world_size · Σ w_t·mean_t / W_global`, so after DDP's mean and the accumulation sum the
   gradient equals the global turn-weighted mean exactly.
6. **Label-position-only LM head.** A wrapper module calls the backbone
   (`model.model(...)`), gathers hidden states at positions whose *next* token is a label,
   and applies the tied `lm_head` only there. The CE is in fp32. DDP wraps the wrapper, so
   gradient hooks still fire.
7. **Per-step control broadcast.** Rank 0 decides checkpoint / eval / stop, using the
   time-based 20-minute trigger, SIGTERM and the session guard. It broadcasts one small
   tensor, so both ranks always enter collectives together.
8. **WSD lr is a stateless function** of `(step, planned_steps, schedule_state)`.
   `schedule_state` (decay start and length, lr scale after NaN recovery) lives in the
   checkpoint. Raising `planned_images` between sessions is therefore safe.
   `--decay-now` sets decay start = resume step, and decay length = min(15% of planned
   steps, the steps that fit in the session at the measured step time).
9. **Storage backends.** `ShardSource` (Hub or local directory) and `Mirror` (Hub or local
   directory) are interfaces. Tests and local smoke runs use directories. Kaggle uses the
   Hub. Fault injection for the flaky-network test wraps either one.
10. **The tokenizer always comes from `Qwen/Qwen3.5-4B`**, even when weights come from
    `unsloth/Qwen3.5-4B` (see the risk above).

## Where each milestone can run

| Milestone | Where | Needs from you |
|---|---|---|
| M0 plan, M1 analysis | here (CPU) | nothing. The parquet (467 MB) and the Qwen3.5 tokenizer files are fetched as the brief requires |
| M2 code, tests, dry run (first ~2,000 members of each tar) + previews | here | nothing |
| M2 full build (stream 117.7 GB from Zenodo, upload ~25–30 GB) | Kaggle CPU notebook or this machine | **HF username + write token, Hub repo visibility, confirmation to start** |
| M3 conversation builder, M4 streaming loader | here | nothing |
| M5a smoke (Qwen3.5-0.8B, 1.75 GB weights) incl. kill→resume | here: RTX 4060 8 GB, `hf` backend, single GPU. DDP logic tested with 2 CPU processes (gloo) on a tiny random model | nothing (0.8B weights are named in the brief) |
| M5b 4B on 2×T4, M6 zero-shot (needs bench shards), M7, M8 | Kaggle | running the notebooks (runbook in `docs/kaggle_runbook.md`) |

Unsloth does not run natively on Windows, so the `unsloth` backend is written to its
documented API and verified only in M5b. `hf` (transformers + PEFT + plain DDP) is fully
tested here and is the fallback.

## Milestones

- **M0**: this plan.
- **M1**: `scripts/analyze_dataset.py` → `reports/dataset_analysis.md`. Reproduce the
  counts, group availability, token lengths, MCQ parse rate, letter/yes-no balance
  train vs bench, malformed rows, and projected tokens per conversation.
- **M2**: `benvqa.data.{quantize,build_shards}`, `scripts/preview_composites.py`,
  `notebooks/kaggle_build_shards.ipynb`. Tests on a synthetic `tar.zst` built in a
  fixture.
- **M3**: `benvqa.data.{mcq,composites,vision,conversation,render}`, with all §6 tests.
- **M4**: `benvqa.data.{stream,collate}` + `scripts/bench_throughput.py`: throughput,
  exact-resume and flaky-network tests.
- **M5**: `benvqa.{train,checkpoint,hub_sync,session_guard,logging_utils}`,
  `benvqa.model.{load,loss}`, configs, notebook. (a) here, (b) Kaggle.
- **M6–M8**: `benvqa.eval_bench`, reports. Code here, runs on Kaggle.

## Risks / open questions

- HF Hub storage for ~25–30 GB of shards. A free account's private storage may not fit,
  so **a public dataset repo is probably required** (the licence allows it with attribution).
- Zenodo throughput and throttling for the 117.7 GB stream. The Range-resume stream
  limits the damage.
- Unsloth + Qwen3.5 vision + `torchrun` DDP + fp16 on Turing is unverified. The `hf`
  backend is the fallback.
- fp16 overflow: weights are published in bf16. The smoke test compares fp16 and fp32
  logits on a few samples.
- `fla` / causal-conv1d kernels on T4: compile time and actual speed-up are measured in M5b.
  Without them the torch fallback is much slower.

## Deviations

(Each entry: what the brief said → what we do → why.)

- Package at `src/benvqa/` instead of `benvqa/`: the repo uses a src layout (see design 1).
- `requests` → `httpx` for streaming: already a dependency, same semantics (design 2).
- The data cursor is counted in micro-batches rather than samples: needed for exact
  resume with length bucketing (design 4). Same guarantee as the brief.
- Group pick probabilities are absolute (binary-yes / binary-no / MCQ = 0.25 / 0.25 / 0.5); an
  unavailable item means the group is skipped for that visit instead of renormalising. M1
  showed 21% of train images have only the adjacency `no` binary; renormalising would
  make adjacency binaries ~2:1 `no`.
- Test split has 115,753 images (the brief's 116,835 included the 1,082 bench images).
