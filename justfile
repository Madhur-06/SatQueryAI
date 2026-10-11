# SatQuery AI task runner. Recipes are single commands so they run under both
# PowerShell (Windows) and sh (Linux/macOS).
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

port := "8000"

# List recipes
default:
    @just --list

# Create .venv with core + dev deps, install git hooks and UI packages
install:
    uv sync
    uv run pre-commit install
    npm --prefix ui install --no-fund --no-audit

# Optional extras (uv sync installs exactly the listed extras, so ml includes eval)
install-eval:
    uv sync --extra eval

# torch (CPU build) + transformers: large download
install-ml:
    uv sync --extra eval --extra ml --extra cpu

# torch (NVIDIA CUDA 12.6 build, driver >= 560) + transformers: larger download
install-gpu:
    uv sync --extra eval --extra ml --extra cuda

# Download a benchmark (verified): just download rsvqa-lr | cdvqa
download NAME:
    uv run satquery-eval --download {{NAME}}

# Lint + format check (Python) and type-check (UI)
lint:
    uv run ruff check .
    uv run ruff format --check .
    npm --prefix ui run typecheck

# Auto-fix lint and format
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Run the test suite (CPU-only)
test *ARGS:
    uv run pytest {{ARGS}}

# Smoke-test the real VLM tool (needs `just install-ml` and cached weights)
test-ml:
    uv run pytest -m ml --run-ml -v

# Build the UI into ui/dist
ui-build:
    npm --prefix ui run build

# Build the UI and start API + UI on http://127.0.0.1:{{port}}
serve: ui-build
    uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port {{port}}

# Development: run these two in separate terminals (UI with hot reload on :5173)
dev-api:
    uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port {{port}} --reload

dev-ui:
    npm --prefix ui run dev

# Write synthetic demo GeoTIFFs to data/demo
demo-data:
    uv run python -m satquery.testing.demo data/demo

# Phase 2.1 (benvqa): extras for the data pipeline + LoRA training on top of install-gpu
install-benvqa:
    uv sync --extra eval --extra ml --extra cuda --extra benvqa --extra train

# Zip committed benvqa code for a Kaggle Dataset ("benvqa-code")
kaggle-code:
    uv run --no-sync python scripts/package_code.py

# Regenerate notebooks/kaggle_*.ipynb from scripts/make_notebooks.py
notebooks:
    uv run --no-sync python scripts/make_notebooks.py
