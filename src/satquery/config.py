"""Runtime settings, read from environment variables prefixed ``SATQUERY_`` or a ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SATQUERY_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    """Root for per-request uploads/outputs (``data_dir/requests/<id>``)."""

    ui_dir: Path = Path("ui/dist")
    """Built UI served at ``/`` when the directory exists (``just ui-build``)."""

    max_upload_mb: int = Field(default=512, gt=0)
    min_overlap_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum footprint IoU for an image pair to be accepted."""

    tile_size: int = Field(default=1024, ge=64)
    tile_overlap: int = Field(default=64, ge=0)

    vlm_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    """Hugging Face model id for the optional zero-shot VLM tool ([ml] extra)."""
    vlm_enabled: bool = False
    """The VLM tool is only offered to the planner when this is explicitly true."""
    vlm_allow_download: bool = False
    """If false, weights must already be in the local Hugging Face cache (no network)."""
    vlm_device: str = "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
