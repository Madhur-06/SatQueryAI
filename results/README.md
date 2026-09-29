# Baselines

Each run writes `<benchmark>_<tool>_<timestamp>.json` (metadata + metrics),
`.records.jsonl.gz` (per-sample predictions) and `.md` (summary). Compare new models
against these rows on the **same subset and seed**.

## RSVQA-LR (test split, 10,004 questions)

| Tool | Subset (seed) | OA | AA | presence | comp | count | rural/urban | Notes |
|---|---|---|---|---|---|---|---|---|
| `vlm_vqa` (mock prior) | all | 39.9 | 30.9 | 75.0 | 32.6 | 15.7 | 0.0 | "yes" / "0" prior; floor |
| `vlm_vqa` (mock prior) | 500 (0) | 42.8 | 32.0 | 79.5 | 30.1 | 18.6 | 0.0 | same subset as below |
| `pipeline` (Qwen off) | 500 (0) | 42.8 | 32.0 | 79.5 | 30.1 | 18.6 | 0.0 | RGB-only: classical tools cannot run, falls back to the mock VLM; 8 area questions rejected |
| `qwen_vl_zeroshot` | 500 (0) | _running_ | | | | | | Qwen/Qwen3-VL-2B-Instruct, CPU, short-answer prompt |

OA = overall accuracy, AA = mean of per-type accuracies (as in the RSVQA papers).

## CDVQA (test split, 39,686 questions)

| Tool | Subset (seed) | OA | AA | Notes |
|---|---|---|---|---|
| `change_vqa` (mock prior) | 1000 (0) | 29.2 | 14.4 | "yes" prior |
| `pipeline` (Qwen off) | 1000 (0) | 29.2 | 14.4 | Otsu-thresholded CVA always finds *some* change, so every yes/no question gets "yes"; categorical answers (`NVG_surface`, `0_to_10`) are not produced |

These classical and mock numbers are floors. The CDVQA result shows that class-specific
change on RGB and CDVQA's answer vocabulary need a learned model or answer adapter
(Phase 2).
