# rsvqa-lr · pipeline

| | |
|---|---|
| Benchmark / split | rsvqa-lr / test |
| Tool | pipeline v0.1.0 (pipeline) |
| Samples (subset, seed) | 500 (500, seed 0) |
| Errors | 8 |
| SatQuery / commit | 0.1.0 / dfa5eb1 |
| Registry digest | 81dda6fb648916d8 |
| Created | 2026-09-29T13:18:00+00:00 |
| Runtime | 8.8 s |

## Metrics

| Metric | Value |
|---|---|
| n | 500 |
| overall_accuracy | 0.4280 |
| average_accuracy | 0.3205 |

### Per question type

| Type | Accuracy | n |
|---|---|---|
| comp | 0.3011 | 186 |
| count | 0.1857 | 140 |
| presence | 0.7952 | 166 |
| rural_urban | 0.0000 | 8 |

## Examples

| Question | Prediction | Reference |
|---|---|---|
| Are there more commercial buildings than large roads? | yes | yes |
| What is the amount of small commercial buildings? | unknown | 11 |
| Are there more grass areas than commercial buildings? | yes | no |
| Is there a wetland? | yes | yes |
| Is a commercial building on the left of a  road present? | yes | yes |
| Is there a rectangular parking? | yes | yes |
| Is the number of water areas equal to the number of farmyards? | 0 | no |
| Are there more commercial buildings than residential buildings? | yes | no |
| What is the amount of grass areas? | ERROR: Cannot map vegetation on optical: the required spectral bands are not in… | 171 |
| Are there less buildings than water areas? | yes | no |
