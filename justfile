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

# Also install the optional extras (ml pulls torch: large download)
install-eval:
    uv sync --extra eval

install-ml:
    uv sync --extra ml

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
