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
| `qwen_vl_zeroshot` | 500 (0) | 47.8 | 33.8 | 63.9 | 71.5 | 0.0 | 0.0 | Qwen/Qwen3-VL-2B-Instruct, CPU (~6.4 s/question). Only ever answers yes/no: count 0 because it never outputs a number; rural/urban 0 because the prompt allows only yes/no or a number (prompt artifact, not re-tuned on test) |

OA = overall accuracy, AA = mean of per-type accuracies (as in the RSVQA papers).

## CDVQA (test split, 39,686 questions)

| Tool | Subset (seed) | OA | AA | Notes |
|---|---|---|---|---|
| `change_vqa` (mock prior) | 1000 (0) | 29.2 | 14.4 | "yes" prior |
| `pipeline` (Qwen off) | 1000 (0) | 29.2 | 14.4 | Otsu-thresholded CVA always finds *some* change, so every yes/no question gets "yes"; categorical answers (`NVG_surface`, `0_to_10`) are not produced |
| `qwen_vl_zeroshot` | 1000 (0) | 42.5 | 24.1 | Qwen/Qwen3-VL-2B-Instruct, both dates as two images, RTX 4060 Laptop GPU (bf16, 0.46 s/question, 7.8 min). Per type: change_or_not 73.4, increase_or_not 64.2, decrease_or_not 55.4; yes/no questions overall 67.9 (always-"yes": 47.1). Categorical and ratio types 0: the prompt allows only yes/no or a number (same artifact as RSVQA rural/urban) |

These classical and mock numbers are floors. The CDVQA result shows that class-specific
change on RGB and CDVQA's answer vocabulary need a learned model or answer adapter
(Phase 2).
