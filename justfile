# SatQuery AI task runner. Recipes are single commands so they run under both
# PowerShell (Windows) and sh (Linux/macOS).
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

port := "8000"

# List recipes
default:
    @just --list

# Create .venv with core + dev deps and install git hooks
install:
    uv sync
    uv run pre-commit install

# Also install the optional extras (ml pulls torch: large download)
install-eval:
    uv sync --extra eval

install-ml:
    uv sync --extra ml

# Lint + format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint and format
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Run the test suite (CPU-only)
test *ARGS:
    uv run pytest {{ARGS}}

# Start the API (serves the built UI at / once `ui/dist` exists)
serve:
    uv run uvicorn satquery.api.app:app --host 127.0.0.1 --port {{port}}
