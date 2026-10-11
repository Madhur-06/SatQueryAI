"""The single training config (YAML -> pydantic). Every tunable number lives here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benvqa.data.conversation import ConversationConfig


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCfg(Strict):
    name: str = "unsloth/Qwen3.5-4B"
    tokenizer: str = "Qwen/Qwen3.5-4B"
    """Always the official tokenizer/processor (``unsloth/`` ships a different one)."""
    backend: Literal["unsloth", "hf"] = "unsloth"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    local_files_only: bool = False


class LoraCfg(Strict):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target: str = "all-linear"
    vision: bool = True


class ImagesCfg(Strict):
    composites: list[str] = Field(default_factory=lambda: ["rgb", "nir_swir", "sar"])
    size: int = 256
    modality_dropout: dict[str, float] = Field(
        default_factory=lambda: {
            "all": 0.60,
            "rgb+nir_swir": 0.15,
            "rgb+sar": 0.10,
            "rgb": 0.10,
            "sar": 0.05,
        }
    )
    composite_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Overrides of ``composites.DEFAULT_COMPOSITES`` (e.g. ``{rgb: {white: [...]}}``)."""


class ConversationCfg(Strict):
    group_pick_probs: dict[str, float] = Field(
        default_factory=lambda: {"binary_yes": 0.25, "binary_no": 0.25, "mcq": 0.5}
    )
    p_drop_presence: float = 0.3
    p_caption: float = 0.2
    caption_turn_weight: float = 1.0
    shuffle_mcq_options: bool = True
    always_relative_pos: bool = True


class ScheduleCfg(Strict):
    type: Literal["wsd", "constant"] = "wsd"
    warmup_steps: int = 100
    decay_fraction: float = 0.15
    min_lr_ratio: float = 0.0


class TrainCfg(Strict):
    micro_batch: int = 2
    grad_accum: int = 8
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    schedule: ScheduleCfg = Field(default_factory=ScheduleCfg)
    planned_images: int = 160_000
    max_grad_norm: float = 1.0
    max_len: int = 1024
    pad_to_multiple: int = 64
    """Pad each micro-batch's length to a multiple (bounded set of kernel shapes, see collate)."""
    pool_size: int = 64
    """Length-bucketing pool (conversations); must be a multiple of ``micro_batch``."""
    nan_patience: int = 3
    max_steps: int | None = None
    """Hard stop (smoke tests); None = run until the session guard or planned images."""
    log_every: int = 10
    ddp_static_graph: bool = False
    """Set if Unsloth's (reentrant) gradient checkpointing trips DDP's ready-twice check."""


class DataCfg(Strict):
    hub_dataset: str = "local:data/benvqa/dev_hub"
    """``hub:<user>/benvqa-shards`` or ``local:<dir>``."""
    revision: str | None = None
    split: str = "train"
    num_workers: int = 2
    interleave: int = 2
    prefetch_shards: int = 2
    prefetch_factor: int = 4
    cache_dir: str = "/kaggle/tmp/shards"
    cache_budget_gb: float = 6.0


class CheckpointCfg(Strict):
    dir: str = "/kaggle/working/ckpt"
    every_minutes: float = 20.0
    every_steps: int = 250
    keep_local: int = 2
    hub_repo: str | None = None
    """``hub:<user>/qwen35-4b-benvqa-lora``, ``local:<dir>`` or None (no mirror)."""
    keep_hub: int = 3


class SessionCfg(Strict):
    max_session_hours: float = 11.3


class EvalCfg(Strict):
    every_steps: int = 1000
    val_questions: int = 2000
    split: str = "val_small"
    batch_size: int = 8
    composites: list[str] | None = None
    """None = all configured composites."""


class Config(Strict):
    seed: int = 1234
    model: ModelCfg = Field(default_factory=ModelCfg)
    lora: LoraCfg = Field(default_factory=LoraCfg)
    images: ImagesCfg = Field(default_factory=ImagesCfg)
    conversation: ConversationCfg = Field(default_factory=ConversationCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    checkpoint: CheckpointCfg = Field(default_factory=CheckpointCfg)
    session: SessionCfg = Field(default_factory=SessionCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    @model_validator(mode="after")
    def _check(self) -> Config:
        for name, value in (
            ("data.hub_dataset", self.data.hub_dataset),
            ("checkpoint.hub_repo", self.checkpoint.hub_repo or ""),
        ):
            if "<HF_USER>" in value:
                raise ValueError(
                    f"{name} still contains <HF_USER>: override it, e.g. "
                    f"--set {name}={value.replace('<HF_USER>', 'yourname')}"
                )
        if self.train.pool_size % self.train.micro_batch:
            raise ValueError("train.pool_size must be a multiple of train.micro_batch")
        self.conversation_config()  # validates probabilities / modality sets
        return self

    def conversation_config(self) -> ConversationConfig:
        return ConversationConfig(
            **self.conversation.model_dump(),
            composites=self.images.composites,
            modality_dropout=self.images.modality_dropout,
        )

    def data_hash(self) -> str:
        """Hash of everything that defines the sample stream (must match to resume cursors)."""
        keys = {
            "seed": self.seed,
            "images": self.images.model_dump(),
            "conversation": self.conversation.model_dump(),
            "micro_batch": self.train.micro_batch,
            "pool_size": self.train.pool_size,
            "max_len": self.train.max_len,
            "data": self.data.model_dump(
                include={"hub_dataset", "revision", "split", "num_workers", "interleave"}
            ),
            "tokenizer": self.model.tokenizer,
        }
        return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:16]


def _set(d: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    """YAML file + ``key.sub=value`` overrides (values parsed as YAML)."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set(raw, key.strip(), yaml.safe_load(value))
    return Config.model_validate(raw)
