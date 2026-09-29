# rsvqa-lr · vlm_vqa

> **Mock tool**: these numbers are a trivial-prior floor, not a model result.

| | |
|---|---|
| Benchmark / split | rsvqa-lr / test |
| Tool | vlm_vqa v0.0.1-mock (tool) |
| Samples (subset, seed) | 10004 (all, seed 0) |
| Errors | 0 |
| SatQuery / commit | 0.1.0 / dfa5eb1 |
| Registry digest | 81dda6fb648916d8 |
| Created | 2026-09-29T13:16:46+00:00 |
| Runtime | 127.7 s |

## Metrics

| Metric | Value |
|---|---|
| n | 10004 |
| overall_accuracy | 0.3985 |
| average_accuracy | 0.3085 |

### Per question type

| Type | Accuracy | n |
|---|---|---|
| comp | 0.3263 | 4002 |
| count | 0.1574 | 2947 |
| presence | 0.7503 | 2955 |
| rural_urban | 0.0000 | 100 |

## Examples

| Question | Prediction | Reference |
|---|---|---|
| Is it a rural or an urban area | yes | urban |
| Is there a grass area? | yes | yes |
| What is the number of roads? | 0 | 403 |
| Is there a road? | yes | yes |
| Is a large road present? | yes | no |
| Are there less buildings than farmlands? | yes | no |
| Is a residential building present? | yes | yes |
| Are there more commercial buildings than roads? | yes | no |
| Is a forest present in the image? | yes | yes |
| What is the amount of farmlands? | unknown | 47 |
