# rsvqa-lr · qwen_vl_zeroshot

| | |
|---|---|
| Benchmark / split | rsvqa-lr / test |
| Tool | qwen_vl_zeroshot v0.1.0 (tool) |
| Samples (subset, seed) | 500 (500, seed 0) |
| Errors | 0 |
| SatQuery / commit | 0.1.0 / fab55cc |
| Registry digest | 81dda6fb648916d8 |
| Created | 2026-09-29T14:27:30+00:00 |
| Runtime | 3219.6 s |

## Metrics

| Metric | Value |
|---|---|
| n | 500 |
| overall_accuracy | 0.4780 |
| average_accuracy | 0.3384 |

### Per question type

| Type | Accuracy | n |
|---|---|---|
| comp | 0.7151 | 186 |
| count | 0.0000 | 140 |
| presence | 0.6386 | 166 |
| rural_urban | 0.0000 | 8 |

## Examples

| Question | Prediction | Reference |
|---|---|---|
| Are there more commercial buildings than large roads? | no | yes |
| What is the amount of small commercial buildings? | yes | 11 |
| Are there more grass areas than commercial buildings? | yes | no |
| Is there a wetland? | yes | yes |
| Is a commercial building on the left of a  road present? | yes | yes |
| Is there a rectangular parking? | yes | yes |
| Is the number of water areas equal to the number of farmyards? | no | no |
| Are there more commercial buildings than residential buildings? | no | no |
| What is the amount of grass areas? | yes | 171 |
| Are there less buildings than water areas? | no | no |
