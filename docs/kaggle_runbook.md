# Kaggle runbook: benvqa (Qwen3.5-4B LoRA on BigEarthNet.txt)

Everything runs from two generated notebooks (`scripts/make_notebooks.py`; do not hand-edit):

| Notebook | Accelerator | Purpose |
|---|---|---|
| `notebooks/kaggle_build_shards.ipynb` | **None** (CPU, costs no GPU quota) | Stage 0: build the ~30 GB shard dataset on the Hub (once) |
| `notebooks/kaggle_train.ipynb` | **GPU T4 x2** | `MODE = smoke` (M5b) / `train` (M7) / `eval` (M6, M8) |

## 0. One-time setup

1. **Hugging Face**: create a token with *write* access (huggingface.co/settings/tokens).
   The notebooks create these repos under your account:
   * `<you>/benvqa-shards` (dataset, ~30 GB). Public by default, because a free account's
     private storage may not fit it. The data licence (CDLA-Permissive-1.0) allows
     redistribution with attribution, and the dataset README carries it. Set
     `PRIVATE = True` in the build notebook if your quota allows.
   * `<you>/qwen35-4b-benvqa-lora` (model, private): checkpoints, `best/`, train log, reports.
   * `<you>/benvqa-smoke` (model, private): only for the M5b round-trip test; delete afterwards.
2. **Kaggle secrets** (Add-ons -> Secrets, attach them to both notebooks): `HF_TOKEN`; optional
   `WANDB_API_KEY`. Add `GITHUB_TOKEN` if the code comes from a private GitHub repo.
3. **Code on Kaggle**, one of:
   * push this repo to GitHub and set `REPO_URL` in the notebook's code cell; or
   * `just kaggle-code` -> `dist/benvqa-code.zip`; upload it as a Kaggle Dataset named
     `benvqa-code` and attach it to the notebooks. Only committed files are packaged.
     Re-upload after code changes. `CODE_VERSION` inside the zip records the commit.
4. Kaggle settings for both notebooks: **Internet on**. Phone verification is required for
   internet and GPUs.

## 1. Build the shards (CPU notebook, once, ~2-4 h)

Open `kaggle_build_shards.ipynb` -> *Save Version -> Save & Run All (Commit)*, accelerator
**None**.
* It streams 117.7 GB from Zenodo over 32 parallel range connections (Zenodo throttles
  each to ~0.4 MB/s) and uploads shard by shard.
* If the session dies or hits 12 h, run it again. Uploaded shards are skipped, and the
  tars are re-streamed but only unfinished patches are decoded.
* The last cell asserts that `meta/shard_index.json` matches the BigEarthNet.txt
  selection (train 229,114 / val_small 3,000 / bench 1,082 images).

## 2. M5b smoke test (GPU notebook, `MODE = "smoke"`, ~30-45 min)

Set `HOURS_PLANNED` = total 2-GPU wall-clock hours you intend to train (30 GPU-h/week =
15 h of 2xT4 per week, so 3 weeks = 45). Run *Save & Run All (Commit)* with **GPU T4 x2**.

Read `reports/m5b_smoke.json` (also printed at the end):

| Check | Expect | If not |
|---|---|---|
| `env.fla_import` | `ok` | fla kernels fail on Turing: training still works (torch fallback, ~1.8x slower). Report it |
| `sweep.<mb>.exit` / `oom` | 0 / false for mb 1 and 2 | keep the largest mb that fits |
| `sweep.<mb>.finite` | true | fp16 overflow in the 4B forward: stop and report (the NaN guard would halve the lr repeatedly) |
| `sweep.<mb>.data_wait_frac` | < 0.05 | loader too slow: `data.num_workers=3` |
| `roundtrip.resumed_from_hub` | true | check the `HF_TOKEN` write scope |
| `recommend.set` | e.g. `train.micro_batch=2 ...` | put these into `EXTRA` for training |

Optional: set `BACKEND = "unsloth"` and run `smoke` again to compare conversations/s.
Unsloth is unverified under DDP on T4 (it doesn't run on Windows, where this was built).
Keep it only if it is faster and `exit` is 0. A backend change means starting a fresh run.

## 3. Zero-shot baselines, M6 (`MODE = "eval"`, `EVAL_RUNS = "a,b"`)

Writes `reports/bench_results.md` (a: RGB only, b: all three composites) and uploads it to
`<you>/qwen35-4b-benvqa-lora/reports/`. Download it from the notebook output.

## 4. Training sessions, M7 (`MODE = "train"`)

* Put the smoke recommendation into `EXTRA`, e.g.
  `EXTRA = ["train.micro_batch=2", "train.grad_accum=8", "train.planned_images=180000"]`.
  Keep the micro-batch/grad-accum settings fixed for the whole run: they are part of the
  data cursor (`data_hash`). `planned_images` may be raised between sessions.
* *Save & Run All (Commit)*, GPU T4 x2. Each session trains up to 11.3 h. The guard
  measures from the notebook's first cell. At the end it evaluates, checkpoints and
  waits for the Hub upload.
* **Continue**: run the same notebook again. It finds `ckpt/LATEST` on the Hub and restores
  adapter, optimizer, scaler, schedule, RNG and the exact data position.
* Checkpoints: every 20 min / 250 steps, the last 2 locally and the last 3 on the Hub, plus
  `best/` by mean(binary OA, MCQ OA) on 2,000 val questions every 1,000 steps.
* Watch progress anywhere via `<you>/qwen35-4b-benvqa-lora/train_log.jsonl` (loss per
  category, lr, conversations/s, `data_wait_frac`, ETA), or W&B if the key is set.
* **Final session**: set `DECAY_NOW = True`. The WSD decay starts at the resume step and
  fits into the session.

Things that stop a resume on purpose: a different number of GPUs, or a changed
data/conversation config (`--allow-data-change` overrides the latter).

## 5. Final evaluation, M8 (`MODE = "eval"`, `EVAL_RUNS = "c,d,e"`)

Uses `<you>/qwen35-4b-benvqa-lora/ckpt/best`. Runs: (c) all composites, (d) RGB only,
(e) SAR only, plus `CAPTIONS` greedy bench captions (BLEU-4/METEOR/ROUGE-L/CIDEr) and
reliability diagrams. All runs (a-e) end up in one `reports/bench_results.md`, next to the
paper's numbers. Each eval session first downloads earlier `reports/` from the Hub repo, so
the table accumulates across sessions.

## Troubleshooting

* `data.hub_dataset still contains <HF_USER>`: the settings cell builds `SETS`; run it.
* `world_size` mismatch on resume: the run must keep using 2 GPUs.
* OOM: lower `train.micro_batch` (and raise `grad_accum` to keep 32 conversations/step);
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set.
* Very slow first steps: the fla Triton kernels compile once per padded sequence length
  (`train.pad_to_multiple: 64` keeps the set small). Expected for the first ~10 steps.
