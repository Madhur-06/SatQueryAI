# SatQuery AI

Agentic vision-language assistant for satellite imagery (SIH 2026, PS SIH26167, ISRO).
Upload one or two GeoTIFFs plus a question. SatQuery validates the inputs, picks tools from a
registry, runs them, and returns a grounded answer with overlays, a confidence score and an
auditable execution trace.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just)
(`uv tool install rust-just`). Node ≥ 20 is needed for the UI.

```sh
just install   # .venv with core + dev deps, git hooks
just lint
just test
just serve     # http://127.0.0.1:8000
```

Optional extras: `just install-eval` (benchmark metrics), `just install-ml` (torch/transformers, large).

See `PROGRESS.md` for status. The architecture and a "how to add a tool" guide land at the end of Phase 1.
