"""Three-channel images built from the 12-channel uint8 cube at load time.

Every display mapping is a fixed per-channel lookup table (no per-image stretching, which
would destroy absolute reflectance / backscatter cues). Optical channels:
``display = (reflectance / white) ** (1 / gamma)``. SAR channels are linear over a dB range.
The third SAR channel is the VV - VH ratio in dB.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from PIL import Image

from benvqa.data.quantize import CHANNELS, QuantSpec, dequantize_s1, dequantize_s2


@dataclass(frozen=True)
class OpticalComposite:
    bands: tuple[str, str, str]
    white: tuple[float, float, float]
    """Reflectance that maps to 255, per channel."""
    gamma: float = 1.8
    kind: Literal["optical"] = "optical"


@dataclass(frozen=True)
class SarComposite:
    vv_db: tuple[float, float] = (-25.0, 0.0)
    vh_db: tuple[float, float] = (-32.0, -5.0)
    ratio_db: tuple[float, float] = (0.0, 15.0)
    kind: Literal["sar"] = "sar"


Composite = OpticalComposite | SarComposite

DEFAULT_COMPOSITES: dict[str, Composite] = {
    "rgb": OpticalComposite(("B04", "B03", "B02"), (0.18, 0.18, 0.18)),
    "nir_swir": OpticalComposite(("B08", "B11", "B12"), (0.50, 0.40, 0.30)),
    "red_edge": OpticalComposite(("B05", "B06", "B07"), (0.25, 0.45, 0.50)),
    "sar": SarComposite(),
}


def composite_from_config(raw: dict) -> Composite:
    raw = dict(raw)
    if raw.pop("kind", "optical") == "sar":
        return SarComposite(**{k: tuple(v) for k, v in raw.items()})
    return OpticalComposite(
        bands=tuple(raw["bands"]), white=tuple(raw["white"]), gamma=raw.get("gamma", 1.8)
    )


@lru_cache(maxsize=64)
def _optical_lut(spec_key: str, band: str, white: float, gamma: float) -> np.ndarray:
    refl = dequantize_s2(np.arange(256), band, QuantSpec.from_json(spec_key))
    out = np.clip(refl / white, 0.0, 1.0) ** (1.0 / gamma)
    return np.rint(out * 255.0).astype(np.uint8)


def _linear_lut(db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.rint(np.clip((db - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


@lru_cache(maxsize=8)
def _sar_luts(spec_key: str, comp: SarComposite) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = QuantSpec.from_json(spec_key)
    codes = np.arange(256)
    vv_db, vh_db = dequantize_s1(codes, "VV", spec), dequantize_s1(codes, "VH", spec)
    ratio = vv_db[:, None] - vh_db[None, :]  # [vv code, vh code]
    return (
        _linear_lut(vv_db, *comp.vv_db),
        _linear_lut(vh_db, *comp.vh_db),
        _linear_lut(ratio, *comp.ratio_db),
    )


_INDEX = {b: i for i, b in enumerate(CHANNELS)}


def make_composite(cube: np.ndarray, comp: Composite, spec: QuantSpec) -> np.ndarray:
    """``[12, H, W]`` uint8 cube -> ``[H, W, 3]`` uint8 image."""
    key = spec.key()
    if isinstance(comp, SarComposite):
        vv, vh = cube[_INDEX["VV"]], cube[_INDEX["VH"]]
        lut_vv, lut_vh, lut_ratio = _sar_luts(key, comp)
        return np.stack([lut_vv[vv], lut_vh[vh], lut_ratio[vv, vh]], axis=-1)
    chans = [
        _optical_lut(key, band, white, comp.gamma)[cube[_INDEX[band]]]
        for band, white in zip(comp.bands, comp.white, strict=True)
    ]
    return np.stack(chans, axis=-1)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Bicubic resize to ``size x size`` (the processor then sees exactly this grid)."""
    if img.shape[:2] == (size, size):
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BICUBIC))
