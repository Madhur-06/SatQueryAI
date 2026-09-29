"""FastAPI application. Milestone 1 only exposes ``/health``; analysis routes arrive in M5."""

from fastapi import FastAPI

from satquery import __version__

app = FastAPI(title="SatQuery AI", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
