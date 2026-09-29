# cdvqa · pipeline

| | |
|---|---|
| Benchmark / split | cdvqa / test |
| Tool | pipeline v0.1.0 (pipeline) |
| Samples (subset, seed) | 1000 (1000, seed 0) |
| Errors | 0 |
| SatQuery / commit | 0.1.0 / dfa5eb1 |
| Registry digest | 81dda6fb648916d8 |
| Created | 2026-09-29T13:28:36+00:00 |
| Runtime | 379.4 s |

## Metrics

| Metric | Value |
|---|---|
| n | 1000 |
| overall_accuracy | 0.2920 |
| average_accuracy | 0.1438 |

### Per question type

| Type | Accuracy | n |
|---|---|---|
| change_or_not | 0.5714 | 399 |
| change_ratio | 0.0000 | 49 |
| change_ratio_types | 0.0000 | 132 |
| change_to_what | 0.0000 | 65 |
| decrease_or_not | 0.2857 | 112 |
| increase_or_not | 0.2936 | 109 |
| largest_change | 0.0000 | 64 |
| smallest_change | 0.0000 | 70 |

## Examples

| Question | Prediction | Reference |
|---|---|---|
| Did the regions of playgrounds change? | yes | no |
| Have the regions of non-vegetated ground surface changed in the second image? | yes | yes |
| Did the regions of trees change? | yes | no |
| Did the areas of water change in the post-change image? | yes | yes |
| Did the regions of non-vegetated ground surface change in the post-event image? | yes | yes |
| Have the regions of water changed? | yes | yes |
| Did the regions of buildings decrease? | yes | no |
| What type of change is the largest in the post-change image? | 60565 | NVG_surface |
| What is the change ratio of trees in the post-change image? | 51031 | 0_to_10 |
| What is the percentage of unchanged areas? | 69487 | 90_to_100 |
