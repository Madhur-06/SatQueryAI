# SatQuery AI — project context (SIH 2026, PS SIH26167, ISRO)

## What we are building
An agentic vision-language assistant for satellite imagery. A user uploads one or two
images plus a natural-language question; the system validates the inputs, picks the
right specialist models/tools from a registry, runs them, fuses their outputs and
returns a grounded answer with visual evidence, a confidence score and an auditable
execution trace.

## Hard requirements from the problem statement
- Inputs: single optical/multispectral or SAR image; co-registered optical+SAR pair;
  bi-temporal pair (same area, two dates). Formats: GeoTIFF/TIFF. PNG/JPEG only for
  public benchmark datasets.
- Must demonstrate: single-image VQA; one more single-image task (captioning or
  text-guided grounding); change description / change-VQA on bi-temporal pairs;
  optical–SAR joint analysis; automatic tool selection and sequencing.

## Engineering conventions
- Python 3.12, `uv` for env/deps, `ruff` (lint+format), `pytest`, `pydantic` v2, type hints everywhere.
- Backend: FastAPI. Geospatial I/O: rasterio (+ numpy, scikit-image).
- All tools implement one `Tool` interface and are registered in the registry;
  the controller never calls a model directly.
- Every tool call is logged to the trace. Every spatial output keeps its geotransform/CRS.
- Heavy dependencies (torch, transformers) are optional extras; the core runs and
  tests pass on a CPU-only laptop.
- Never download datasets or model weights without asking first. Never invent URLs or
  Hugging Face IDs — verify them or leave a TODO.
- Tests use small synthetic GeoTIFFs generated in fixtures, not real data.
- Keep PROGRESS.md updated: what's done, what's next, open questions.
