"""Zero-shot VQA / captioning / change-VQA with a Qwen-VL model (``[ml]`` extra).

The model id comes from settings (default ``Qwen/Qwen3-VL-2B-Instruct``, verified on
Hugging Face, Apache-2.0, needs transformers >= 4.57). Nothing is imported or loaded
until the first ``run``. The tool is only *available* when ``SATQUERY_VLM_ENABLED=true``
and torch + transformers are installed. Weights are read from the local cache unless
``SATQUERY_VLM_ALLOW_DOWNLOAD=true``.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any, Literal

from pydantic import Field

from satquery.config import get_settings
from satquery.io.preprocess.optical import false_colour
from satquery.io.render import quicklook
from satquery.io.validate import RasterLayer
from satquery.tools.base import Tool, ToolInput, ToolOutput, ToolParams, ToolRequirements
from satquery.types import Capability, InputConfig, TaskType

PROMPTS = {
    "vqa": "You are analysing a satellite image. Answer the question concisely. {question}",
    "caption": "Describe this satellite image in two or three sentences: land cover, "
    "notable structures, water bodies and patterns.",
    "change_vqa": "The first image was acquired before the second image; both show the same "
    "area. Answer concisely. {question}",
}


class QwenVLParams(ToolParams):
    mode: Literal["vqa", "caption", "change_vqa"] = "vqa"
    rendering: Literal["true_colour", "false_colour"] = "true_colour"
    max_new_tokens: int = Field(default=128, ge=1, le=1024)


class QwenVLZeroShot(Tool):
    name = "qwen_vl_zeroshot"
    version = "0.1.0"
    description = (
        "Zero-shot Qwen-VL vision-language model (model id from settings) for VQA, "
        "captioning and two-image change VQA. Optional: needs the [ml] extra."
    )
    tasks = frozenset({TaskType.VQA, TaskType.CAPTION, TaskType.CHANGE_VQA, TaskType.CHANGE_DESC})
    input_configs = frozenset(InputConfig)
    produces = frozenset({Capability.ANSWER, Capability.CAPTION, Capability.CHANGE_ANSWER})
    requires = ToolRequirements(n_layers=1)
    Params = QwenVLParams
    priority = 20  # preferred over mocks whenever it is available

    _lock = threading.Lock()
    _model: Any = None
    _processor: Any = None

    def is_available(self) -> tuple[bool, str | None]:
        settings = get_settings()
        if not settings.vlm_enabled:
            return False, "disabled (set SATQUERY_VLM_ENABLED=true)"
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"missing packages {missing} (install the [ml] extra)"
        return True, None

    def _load(self) -> tuple[Any, Any]:
        with self._lock:
            if QwenVLZeroShot._model is None:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                settings = get_settings()
                local_only = not settings.vlm_allow_download
                QwenVLZeroShot._processor = AutoProcessor.from_pretrained(
                    settings.vlm_model_id, local_files_only=local_only
                )
                QwenVLZeroShot._model = AutoModelForImageTextToText.from_pretrained(
                    settings.vlm_model_id,
                    dtype="auto",
                    device_map=settings.vlm_device,
                    local_files_only=local_only,
                )
        return QwenVLZeroShot._model, QwenVLZeroShot._processor

    @staticmethod
    def _image(layer: RasterLayer, rendering: str):
        from PIL import Image

        rgb = None
        if rendering == "false_colour":
            rgb = false_colour(layer.data, layer.band_map, layer.valid)
        if rgb is None:
            rgb = quicklook(layer.data, layer.valid, layer.modality, layer.band_map)
        return Image.fromarray(rgb)

    def run(self, inp: ToolInput, params: QwenVLParams) -> ToolOutput:
        model, processor = self._load()
        n_images = 2 if params.mode == "change_vqa" and len(inp.layers) > 1 else 1
        images = [self._image(inp.layer(i), params.rendering) for i in range(n_images)]
        prompt = PROMPTS[params.mode].format(question=inp.query or "")
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=params.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return ToolOutput(
            text=text,
            values={"answer": text, "model_id": get_settings().vlm_model_id},
            confidence=None,
            confidence_basis="zero-shot generation (uncalibrated)",
        )
