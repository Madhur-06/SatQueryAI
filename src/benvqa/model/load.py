"""Load Qwen3.5 + a 16-bit LoRA adapter (backends: ``hf`` = transformers + PEFT, ``unsloth``).

* Base weights in ``model.dtype`` (fp16 on T4), LoRA weights in fp32 (GradScaler-safe).
* LoRA on every linear layer of the language model and (``lora.vision``) the vision tower.
  ``lm_head`` / ``embed_tokens`` are tied (636 M params) and are never trained or saved.
  The MTP head is not even loaded by transformers.
* The tokenizer/processor always comes from ``model.tokenizer`` (official Qwen repo).

Unsloth API as documented on unsloth.ai/docs/models/qwen3.5/fine-tune (2026-10):
``FastVisionModel.from_pretrained(..., load_in_4bit=False, load_in_16bit=True)`` and
``FastVisionModel.get_peft_model(..., finetune_vision_layers=..., target_modules="all-linear")``.
That path is only exercised on Kaggle (M5b): Unsloth does not run natively on Windows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from benvqa.config import Config

log = logging.getLogger(__name__)
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VISION_EXCLUDE = r".*\.visual\..*"


@dataclass
class Loaded:
    model: Any
    """PEFT-wrapped model (trainable LoRA only)."""
    base: Any
    """The underlying ``Qwen3_5ForConditionalGeneration`` (``.model``, ``.lm_head``)."""
    backend: str
    info: dict[str, Any] = field(default_factory=dict)


def unwrap_base(model: Any) -> Any:
    """DDP / loss wrapper / PEFT -> the ``*ForConditionalGeneration`` module.

    (``hasattr`` is useless here: PEFT forwards unknown attributes to the base model.)
    """
    m = model
    while True:
        if isinstance(m, torch.nn.parallel.DistributedDataParallel):
            m = m.module
        elif hasattr(type(m), "get_base_model"):
            return m.get_base_model()
        elif type(m).__name__ == "LabelPositionLM":
            m = m.model
        else:
            return m


def _upcast_trainable(model: Any) -> tuple[int, int]:
    trainable = total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            p.data = p.data.float()
            trainable += p.numel()
    return trainable, total


def _check_no_tied_training(model: Any) -> None:
    for name, p in model.named_parameters():
        if p.requires_grad and ("lm_head" in name or "embed_tokens" in name or "mtp" in name):
            raise AssertionError(f"{name} must not be trainable")


def load_hf(cfg: Config, device: torch.device) -> Loaded:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.name,
        dtype=DTYPES[cfg.model.dtype],
        device_map={"": device},
        attn_implementation=cfg.model.attn_implementation,
        local_files_only=cfg.model.local_files_only,
    )
    model.config.use_cache = False
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target,
        exclude_modules=None if cfg.lora.vision else VISION_EXCLUDE,
        bias="none",
    )
    peft_model = get_peft_model(model, lcfg)
    trainable, total = _upcast_trainable(peft_model)
    _check_no_tied_training(peft_model)
    return Loaded(
        peft_model,
        unwrap_base(peft_model),
        "hf",
        {"load_s": round(time.time() - t0, 1), "trainable": trainable, "total": total},
    )


def load_unsloth(cfg: Config, device: torch.device) -> Loaded:
    from unsloth import FastVisionModel

    t0 = time.time()
    model, _ = FastVisionModel.from_pretrained(
        cfg.model.name,
        load_in_4bit=False,
        load_in_16bit=True,
        dtype=DTYPES[cfg.model.dtype],
        device_map={"": device.index if device.type == "cuda" else "cpu"},
        use_gradient_checkpointing="unsloth" if cfg.model.gradient_checkpointing else False,
        full_finetuning=False,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.lora.vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        random_state=cfg.seed,
        target_modules=cfg.lora.target,
    )
    FastVisionModel.for_training(model)
    trainable, total = _upcast_trainable(model)
    _check_no_tied_training(model)
    return Loaded(
        model,
        unwrap_base(model),
        "unsloth",
        {"load_s": round(time.time() - t0, 1), "trainable": trainable, "total": total},
    )


def use_reference_kernels() -> list[str]:
    """Route GatedDeltaNet / causal-conv calls to transformers' torch reference code.

    transformers binds the fla / causal_conv1d implementations at import time whenever the
    packages are installed; those are GPU-only (Triton/CUDA), so CPU runs must switch back.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    swapped = []
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):  # fmt: skip
        fn = getattr(qwen, name, None)
        ref = getattr(fn, "__wrapped__", None)
        if ref is not None:
            setattr(qwen, name, ref)
            swapped.append(name)
    return swapped


def load_model(cfg: Config, device: torch.device) -> Loaded:
    if device.type != "cuda":
        use_reference_kernels()
    loaded = load_unsloth(cfg, device) if cfg.model.backend == "unsloth" else load_hf(cfg, device)
    loaded.info["fast_kernels"] = fast_kernel_report()
    log.info("loaded %s (%s): %s", cfg.model.name, loaded.backend, loaded.info)
    return loaded


def fast_kernel_report() -> dict[str, bool]:
    """Whether the fast GatedDeltaNet / causal-conv kernels are importable (else torch fallback)."""
    import importlib.util

    return {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "triton")}


def load_adapter_weights(model: Any, state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, state)
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise KeyError(f"unexpected adapter keys: {unexpected[:5]}")


def adapter_state(model: Any) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    return {
        k: v.detach().to("cpu", torch.float32).contiguous()
        for k, v in get_peft_model_state_dict(model).items()
    }
