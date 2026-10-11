"""Synthetic stand-ins for BigEarthNet.txt + BigEarthNet v2 archives (tests only, no real data).

* ``synth_rows``: annotation rows with the real per-image structure and string formats
  (2 binaries + 1 MCQ for presence/area/count, adjacency `no` always and `yes`/MCQ only
  when a touching pair exists, optional relative position, country/season/climate MCQs
  with comma-containing options, a caption, bbox rows).
* ``write_parquet``: a ``BigEarthNet.txt.parquet`` with the real columns.
* ``write_tars``: ``BigEarthNet-S{1,2}.tar.zst`` with the real member layout, band shapes
  and dtypes (S2 uint16, 20 m bands at 60x60; S1 float32 dB).
* ``RangeHTTPServer``: serves files with HTTP Range support and can drop connections after
  given byte counts (resumable-stream tests).
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = [
    "Arable land",
    "Coniferous forest",
    "Urban fabric",
    "Pastures",
    "Inland waters",
    "Broad-leaved forest",
    "Mixed forest",
    "Complex cultivation patterns",
]
COUNTRIES = ["Finland", "Portugal", "Serbia", "Austria", "Lithuania", "Ireland"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]
CLIMATES = [
    "Cold, no dry season, warm summer",
    "Temperate, dry summer, hot summer",
    "Cold, no dry season, cold summer",
    "Temperate, no dry season, warm summer",
]
S2_ALL = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
S2_SHAPE = {"B01": 20, "B09": 20, "B05": 60, "B06": 60, "B07": 60, "B8A": 60, "B11": 60, "B12": 60}


def _mcq(rng: np.random.Generator, question: str, correct: str, pool: list[str]) -> tuple[str, str]:
    distract = [x for x in pool if x != correct]
    opts = [correct, *rng.choice(distract, size=3, replace=False).tolist()]
    order = rng.permutation(4)
    opts = [opts[i] for i in order]
    letter = "abcd"[int(np.flatnonzero(order == 0)[0])]
    text = question + " " + ", ".join(f"{'abcd'[i]}) {o}" for i, o in enumerate(opts))
    return text, letter


def synth_rows(
    patch_id: str,
    s1_name: str,
    split: str,
    rng: np.random.Generator,
    *,
    adjacency: bool = True,
    relpos: bool = False,
    start_id: int = 0,
) -> list[dict[str, Any]]:
    present = rng.choice(CLASSES, size=3, replace=False).tolist()
    absent = [c for c in CLASSES if c not in present]
    country, season = str(rng.choice(COUNTRIES)), str(rng.choice(SEASONS))
    climate = str(rng.choice(CLIMATES))
    rows: list[tuple[str, str, str, str]] = []  # type, category, input, output
    a, b = present[0].lower(), present[1].lower()
    rows += [
        ("binary", "presence", f"Is there any {a} in the image?", "yes"),
        ("binary", "presence", f"Can any {absent[0].lower()} be observed in this image?", "no"),
        (
            "mcq",
            "presence",
            *_mcq(rng, "Which class appears in the image?", present[0], [*absent, present[0]]),
        ),
        ("binary", "area", f"Does {a} cover at least 10% of the image?", "yes"),
        ("binary", "area", f"Is the total area of {b} between 60% and 90%?", "no"),
        (
            "mcq",
            "area",
            *_mcq(
                rng,
                f"How much of the image is covered by {a}?",
                "20 to 40%",
                ["0 to 20%", "20 to 40%", "40 to 60%", "60 to 100%", "80 to 90%"],
            ),
        ),
        ("binary", "count", f"Is there only one continuous patch of {a}?", "yes"),
        ("binary", "count", f"Are there at least four separate areas of {b}?", "no"),
        (
            "mcq",
            "count",
            *_mcq(
                rng,
                f"How many distinct regions of {b} are there?",
                "2",
                ["0", "1", "2", "3", "4", "More than five"],
            ),
        ),
        ("binary", "adjacency", f"Does any {a} touch {absent[1].lower()}?", "no"),
        ("mcq", "country", *_mcq(rng, "Which country is shown?", country, COUNTRIES)),
        ("mcq", "season", *_mcq(rng, "In which season was the image acquired?", season, SEASONS)),
        ("mcq", "climate zone", *_mcq(rng, "Select the climate zone:", climate, CLIMATES)),
        (
            "captioning",
            "None",
            "Describe the image.",
            f"This image from {country} in {season.lower()} shows {a} next to {b}, in a "
            f"'{climate}' climate.",
        ),
        (
            "bounding box",
            "reference",
            f"Point out the <ref>largest area of {a}</ref>.",
            "[0.1 0.2, 0.5 0.9]",
        ),
    ]
    if adjacency:
        rows += [
            ("binary", "adjacency", f"Does any {a} border {b}?", "yes"),
            (
                "mcq",
                "adjacency",
                *_mcq(
                    rng,
                    "Which pair is touching?",
                    f"{present[0]} and {present[1]}",
                    [f"{x} and {y}" for x in CLASSES[:4] for y in CLASSES[4:]]
                    + [f"{present[0]} and {present[1]}"],
                ),
            ),
        ]
    if relpos:
        rows.append(
            (
                "mcq",
                "relative pos",
                *_mcq(
                    rng,
                    f"Where is the {a} relative to the {b}?",
                    "to the top-left",
                    ["to the top-left", "to the right", "to the bottom", "to the top-right"],
                ),
            )
        )
    return [
        {
            "ID": start_id + i,
            "s1_name": s1_name,
            "patch_id": patch_id,
            "input": inp,
            "output": out,
            "type": t,
            "category": c,
            "split": split,
            "latitude": 50.0,
            "longitude": 10.0,
            "country": country,
            "season": season,
            "climate_zone": climate,
        }
        for i, (t, c, inp, out) in enumerate(rows)
    ]


def patch_names(tile: int, i: int) -> tuple[str, str]:
    s2 = f"S2A_MSIL2A_20170613T101031_N9999_R022_T{tile:02d}UUP_{i:02d}_{i + 1:02d}"
    s1 = f"S1A_IW_GRDH_1SDV_20170613T165043_{tile:02d}UUP_{60 + i:02d}_{40 + i:02d}"
    return s2, s1


def make_images(n_per_split: dict[str, int], seed: int = 0) -> list[dict[str, Any]]:
    """Image descriptors spread over several tiles (tars are ordered by tile)."""
    rng = np.random.default_rng(seed)
    out, k = [], 0
    for split, n in n_per_split.items():
        for _ in range(n):
            s2, s1 = patch_names(tile=k % 4, i=k)
            out.append(
                {
                    "patch_id": s2,
                    "s1_name": s1,
                    "split": split,
                    "adjacency": bool(rng.random() < 0.8),
                    "relpos": bool(rng.random() < 0.3),
                }
            )
            k += 1
    return out


def write_parquet(path: Path, images: list[dict[str, Any]], seed: int = 0) -> Path:
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for img in images:
        rows += synth_rows(
            img["patch_id"],
            img["s1_name"],
            img["split"],
            rng,
            adjacency=img["adjacency"],
            relpos=img["relpos"],
            start_id=len(rows),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _tif_bytes(arr: np.ndarray) -> bytes:
    import tifffile

    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return buf.getvalue()


def _seed(*parts: str) -> int:
    return int.from_bytes(
        hashlib.blake2b("|".join(parts).encode(), digest_size=4).digest(), "little"
    )


def s2_band(patch_id: str, band: str) -> np.ndarray:
    """Deterministic DN pattern for a patch/band (tests compare decoded cubes to it)."""
    n = S2_SHAPE.get(band, 120)
    seed = _seed(patch_id, band)
    base = 300 + 200 * S2_ALL.index(band)
    rng = np.random.default_rng(seed)
    return (base + rng.integers(0, 400, (n, n))).astype(np.uint16)


def s1_band(s1_name: str, band: str) -> np.ndarray:
    seed = _seed(s1_name, band)
    centre = -12.0 if band == "VV" else -19.0
    return (centre + np.random.default_rng(seed).normal(0, 3, (120, 120))).astype(np.float32)


def write_tars(root: Path, images: list[dict[str, Any]], extra: int = 3) -> tuple[Path, Path]:
    """Write ``BigEarthNet-S2.tar.zst`` / ``-S1.tar.zst``, sorted by tile like the real ones,
    plus ``extra`` patches that are not in the annotations."""
    import zstandard

    root.mkdir(parents=True, exist_ok=True)
    all_imgs = list(images) + [
        dict(zip(("patch_id", "s1_name"), patch_names(tile=9, i=90 + j), strict=True))
        for j in range(extra)
    ]
    paths = []
    for sensor in ("S2", "S1"):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as tf:
            key = "patch_id" if sensor == "S2" else "s1_name"
            for img in sorted(all_imgs, key=lambda d: d[key]):
                name = img[key]
                folder = name.rsplit("_", 2)[0]
                bands = S2_ALL if sensor == "S2" else ["VH", "VV"]
                for band in sorted(bands):
                    arr = s2_band(name, band) if sensor == "S2" else s1_band(name, band)
                    data = _tif_bytes(arr)
                    info = tarfile.TarInfo(
                        f"BigEarthNet-{sensor}/{folder}/{name}/{name}_{band}.tif"
                    )
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        out = root / f"BigEarthNet-{sensor}.tar.zst"
        out.write_bytes(zstandard.ZstdCompressor(level=3).compress(raw.getvalue()))
        paths.append(out)
    return paths[0], paths[1]


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args: object) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        data = self.server.files.get(self.path.lstrip("/"))
        if data is None:
            self.send_error(404)
            return
        start, stop = 0, len(data)  # stop is exclusive
        rng = self.headers.get("Range")
        if rng:
            first, _, last = rng.removeprefix("bytes=").partition("-")
            start = int(first)
            stop = min(len(data), int(last) + 1) if last else len(data)
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(stop - start))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with self.server.lock:
            self.server.requests += 1
            drop = self.server.drops.pop(0) if self.server.drops else None
        end = stop if drop is None else min(stop, start + drop)
        try:
            self.wfile.write(data[start:end])
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        if drop is not None:
            self.close_connection = True  # truncated body: the client must resume


class _Server(ThreadingHTTPServer):
    files: dict[str, bytes]
    drops: list[int]
    requests: int
    lock: threading.Lock


@contextmanager
def range_server(files: dict[str, Path], drops: list[int] | None = None) -> Iterator[_Server]:
    """Serve ``{url_path: file}`` on localhost. ``drops``: per request, close the
    connection after that many body bytes (consumed in order)."""
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.files = {k: Path(v).read_bytes() for k, v in files.items()}
    srv.drops = list(drops or [])
    srv.requests = 0
    srv.lock = threading.Lock()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def url_of(srv: _Server, name: str) -> str:
    host, port = srv.server_address[:2]
    return f"http://{host}:{port}/{name}"


class FakeTokenizer:
    """Deterministic word-level stand-in for the Qwen tokenizer (stream/trainer tests).

    Special tokens get fixed ids; every other word / punctuation / whitespace run is hashed
    to an id in ``[1000, vocab_size)``. Picklable, so DataLoader workers can use it.
    """

    SPECIALS = (
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<think>",
        "</think>",
    )

    def __init__(self, vocab_size: int = 4096) -> None:
        import re

        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self._split = re.compile("(" + "|".join(re.escape(s) for s in self.SPECIALS) + ")")
        self._words = re.compile(r"\w+|[^\w\s]|\s+")
        self._pieces: dict[int, str] = {}

    def _id(self, piece: str) -> int:
        i = 1000 + _seed(piece) % (self.vocab_size - 1000)
        self._pieces.setdefault(i, piece)
        return i

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out = []
        for i in ids:
            if i < len(self.SPECIALS):
                if not skip_special_tokens:
                    out.append(self.SPECIALS[i])
            else:
                out.append(self._pieces.get(i, f"<{i}>"))
        return "".join(out)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids: list[int] = []
        for part in self._split.split(text):
            if part in self.SPECIALS:
                ids.append(self.SPECIALS.index(part))
            elif part:
                ids += [self._id(w) for w in self._words.findall(part)]
        return {"input_ids": ids}

    def __getstate__(self) -> dict:
        return {"vocab_size": self.vocab_size}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["vocab_size"])


def write_tiny_model(path: Path, seed: int = 0) -> Path:
    """A randomly initialised, tiny Qwen3.5 VLM (same architecture: GatedDeltaNet + full
    attention, vision tower, M-RoPE, tied embeddings) whose special token ids match
    ``FakeTokenizer``. Trainer tests run on CPU in seconds with it."""
    import torch
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5 import Qwen3_5Config

    torch.manual_seed(seed)
    tok = FakeTokenizer()
    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": tok.vocab_size,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "tie_word_embeddings": True,
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "pad_token_id": 0,
            "eos_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 2304,
        },
        image_token_id=FakeTokenizer.SPECIALS.index("<|image_pad|>"),
        video_token_id=999,
        vision_start_token_id=FakeTokenizer.SPECIALS.index("<|vision_start|>"),
        vision_end_token_id=FakeTokenizer.SPECIALS.index("<|vision_end|>"),
        tie_word_embeddings=True,
    )
    model = AutoModelForImageTextToText.from_config(cfg)
    model.save_pretrained(path)
    return path
