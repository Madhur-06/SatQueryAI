"""12-channel uint8 encoding of a BigEarthNet v2 patch (S2 10 bands + S1 VV/VH).

Verified on the Zenodo archive (M2 dry run, see ``docs/phase2_vqa_plan.md``):

* S2 L2A tiles are uint16 DN with **no** +1000 BOA offset (processing baseline ``N9999``;
  dark-water B02 minima are 70-90 DN). Reflectance = DN / 10000.
* 10 m bands are 120x120; 20 m bands (B05 B06 B07 B8A B11 B12) are 60x60 and are upsampled
  bilinearly (pixel-centre aligned) to 120x120. B01 and B09 (60 m) are dropped.
* S1 tiles are float32 backscatter **already in dB**.

Encoding: S2 ``u8 = round(255 * sqrt(clip(refl / max_b, 0, 1)))`` (sqrt keeps precision
for dark targets), S1 linear over a fixed dB range per polarisation. The parameters are
stored in ``meta/quantization.json`` next to the shards; decoding must use the same ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

S2_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
S2_20M: frozenset[str] = frozenset({"B05", "B06", "B07", "B8A", "B11", "B12"})
S1_BANDS: tuple[str, ...] = ("VV", "VH")
CHANNELS: tuple[str, ...] = S2_BANDS + S1_BANDS
SIZE = 120
CUBE_SHAPE = (len(CHANNELS), SIZE, SIZE)
CUBE_BYTES = int(np.prod(CUBE_SHAPE))


def _default_s2_max() -> dict[str, float]:
    return {
        "B02": 0.3,
        "B03": 0.3,
        "B04": 0.3,
        "B05": 0.6,
        "B06": 0.6,
        "B07": 0.6,
        "B08": 0.6,
        "B8A": 0.6,
        "B11": 0.5,
        "B12": 0.5,
    }


def _default_s1_range() -> dict[str, tuple[float, float]]:
    return {"VV": (-30.0, 5.0), "VH": (-40.0, 0.0)}


@dataclass(frozen=True)
class QuantSpec:
    version: int = 1
    s2_offset: float = 0.0
    s2_scale: float = 10000.0
    s2_max: dict[str, float] = field(default_factory=_default_s2_max)
    s1_units: str = "dB"
    s1_range_db: dict[str, tuple[float, float]] = field(default_factory=_default_s1_range)
    channels: tuple[str, ...] = CHANNELS
    shape: tuple[int, int, int] = CUBE_SHAPE
    upsample: str = "bilinear, pixel-centre aligned (20 m -> 10 m)"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> QuantSpec:
        raw = json.loads(text)
        raw["s1_range_db"] = {k: tuple(v) for k, v in raw["s1_range_db"].items()}
        raw["channels"] = tuple(raw["channels"])
        raw["shape"] = tuple(raw["shape"])
        return cls(**raw)

    def key(self) -> str:
        return self.to_json()


@lru_cache(maxsize=64)
def _s2_lut(spec_key: str, band: str) -> np.ndarray:
    spec = QuantSpec.from_json(spec_key)
    dn = np.arange(65536, dtype=np.float64)
    refl = (dn - spec.s2_offset) / spec.s2_scale
    u8 = np.rint(255.0 * np.sqrt(np.clip(refl / spec.s2_max[band], 0.0, 1.0)))
    return u8.astype(np.uint8)


def quantize_s2(dn: np.ndarray, band: str, spec: QuantSpec) -> np.ndarray:
    """uint16 DN -> uint8 code via a 65,536-entry lookup table."""
    return _s2_lut(spec.key(), band)[np.asarray(dn, dtype=np.uint16)]


def dequantize_s2(u8: np.ndarray, band: str, spec: QuantSpec) -> np.ndarray:
    """uint8 code -> surface reflectance (float32)."""
    frac = (np.asarray(u8, dtype=np.float32) / 255.0) ** 2
    return frac * np.float32(spec.s2_max[band])


def quantize_s1(db: np.ndarray, band: str, spec: QuantSpec) -> np.ndarray:
    lo, hi = spec.s1_range_db[band]
    x = np.nan_to_num(np.asarray(db, dtype=np.float32), nan=lo, posinf=hi, neginf=lo)
    return np.rint(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def dequantize_s1(u8: np.ndarray, band: str, spec: QuantSpec) -> np.ndarray:
    lo, hi = spec.s1_range_db[band]
    return np.float32(lo) + np.asarray(u8, dtype=np.float32) / 255.0 * np.float32(hi - lo)


@lru_cache(maxsize=16)
def _interp_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Bilinear weights with pixel centres aligned (``align_corners=False``), edges clamped."""
    pos = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
    pos = np.clip(pos, 0, n_in - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (pos - lo).astype(np.float32)
    m = np.zeros((n_out, n_in), dtype=np.float32)
    m[np.arange(n_out), lo] += 1.0 - w
    m[np.arange(n_out), hi] += w
    return m


def upsample(a: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Separable bilinear resize of a 2-D array to ``size x size`` (float32 output)."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape == (size, size):
        return a
    return _interp_matrix(a.shape[0], size) @ a @ _interp_matrix(a.shape[1], size).T


def encode_cube(s2: dict[str, np.ndarray], s1_u8: np.ndarray | None, spec: QuantSpec) -> np.ndarray:
    """Build the ``[12, 120, 120]`` uint8 cube from raw S2 DNs and the already quantized
    ``[2, 120, 120]`` S1 codes (VV, VH). ``s1_u8=None`` leaves VV/VH at zero."""
    cube = np.zeros(CUBE_SHAPE, dtype=np.uint8)
    for i, band in enumerate(S2_BANDS):
        dn = s2[band]
        if dn.shape != (SIZE, SIZE):
            dn = np.clip(np.rint(upsample(dn)), 0, 65535).astype(np.uint16)
        cube[i] = quantize_s2(dn, band, spec)
    if s1_u8 is not None:
        cube[len(S2_BANDS) :] = s1_u8
    return cube


def cube_from_bytes(buf: bytes | memoryview) -> np.ndarray:
    """Zero-copy view of a stored cube."""
    return np.frombuffer(buf, dtype=np.uint8).reshape(CUBE_SHAPE)
