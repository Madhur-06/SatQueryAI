"""Score-based single-turn VQA evaluation (no generation), shared by training and bench eval.

Each question is rendered as the single-turn inference prompt (system + tagged images +
question + non-thinking assistant prefix). One forward pass reads the logits at the last
prompt position over the candidate answers actually offered (``yes``/``no`` or the MCQ
letters). The prediction is the argmax, and the softmax over candidates is the confidence
(reused in Phase 3). Inputs are used verbatim: no option shuffling.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from benvqa.data.collate import padded_length
from benvqa.data.conversation import QAItem
from benvqa.data.mcq import parse_mcq
from benvqa.data.render import Renderer
from benvqa.data.stream import ImageMaker, bands_array
from benvqa.hub_sync import Mirror

PAPER_BINARY = ("presence", "area", "count", "adjacency")
PAPER_MCQ = (
    "presence",
    "area",
    "count",
    "adjacency",
    "relative pos",
    "country",
    "season",
    "climate zone",
)
ABBREV = {
    "presence": "Pr",
    "area": "A",
    "count": "Cnt",
    "adjacency": "Adj",
    "relative pos": "RP",
    "country": "Loc",
    "season": "S",
    "climate zone": "Clt",
}


@dataclass
class EvalQuestion:
    qid: int
    patch_id: str
    cube: np.ndarray
    question: str
    answer: str
    type: str
    category: str
    candidates: tuple[str, ...]


def load_split(mirror: Mirror, cache_dir: Path, split: str) -> list[dict[str, Any]]:
    """All rows of a shard split: patch_id, cube view, qa items."""
    rows: list[dict[str, Any]] = []
    for rel in sorted(p for p in mirror.list(f"{split}/") if p.endswith(".parquet")):
        table = pq.read_table(
            mirror.download(rel, cache_dir), columns=["patch_id", "split", "bands", "qa"]
        )
        assert set(table.column("split").to_pylist()) == {split}
        cubes = bands_array(table)
        for i, (pid, qa) in enumerate(
            zip(table.column("patch_id").to_pylist(), table.column("qa").to_pylist(), strict=True)
        ):
            rows.append(
                {"patch_id": pid, "cube": cubes[i], "qa": [QAItem(**d) for d in json.loads(qa)]}
            )
    return rows


def vqa_questions(rows: list[dict[str, Any]]) -> list[EvalQuestion]:
    out = []
    for row in rows:
        for it in row["qa"]:
            if it.type == "binary":
                cands: tuple[str, ...] = ("yes", "no")
            elif it.type == "mcq":
                parsed = parse_mcq(it.input)
                cands = tuple(parsed.letters) if parsed else ("a", "b", "c", "d")
            else:
                continue
            out.append(
                EvalQuestion(
                    it.id,
                    row["patch_id"],
                    row["cube"],
                    it.input,
                    it.output,
                    it.type,
                    it.category,
                    cands,
                )
            )
    return out


def balanced_subset(questions: list[EvalQuestion], n: int, seed: int = 0) -> list[EvalQuestion]:
    """Up to ``n`` questions spread evenly over (type, category) cells (fixed seed)."""
    cells: dict[tuple[str, str], list[EvalQuestion]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.qid):
        cells[(q.type, q.category)].append(q)
    rng = random.Random(seed)
    for v in cells.values():
        rng.shuffle(v)
    out: list[EvalQuestion] = []
    keys = sorted(cells)
    i = 0
    while len(out) < n and any(cells.values()):
        k = keys[i % len(keys)]
        if cells[k]:
            out.append(cells[k].pop())
        i += 1
    return sorted(out, key=lambda q: q.qid)


class ScoreEvaluator:
    def __init__(
        self,
        renderer: Renderer,
        images: ImageMaker,
        composites: tuple[str, ...],
        batch_size: int = 8,
        pad_to_multiple: int = 64,
    ) -> None:
        self.renderer, self.images = renderer, images
        self.composites, self.batch_size = composites, batch_size
        self.pad_to_multiple = pad_to_multiple
        self._cand_ids: dict[str, int] = {}

    def cand_id(self, word: str) -> int:
        if word not in self._cand_ids:
            ids = self.renderer.encode(word)
            assert len(ids) == 1, f"candidate {word!r} is not one token"
            self._cand_ids[word] = ids[0]
        return self._cand_ids[word]

    def batch(self, qs: list[EvalQuestion]) -> dict[str, Any]:
        import torch

        prompts = [self.renderer.prompt_ids(q.question, self.composites) for q in qs]
        L = padded_length(max(map(len, prompts)), self.pad_to_multiple)
        ids = np.full((len(qs), L), self.renderer.pad_id, dtype=np.int64)
        att = np.zeros_like(ids)
        tiles, grids = [], []
        for b, (q, p) in enumerate(zip(qs, prompts, strict=True)):
            ids[b, : len(p)] = p
            att[b, : len(p)] = 1
            t, g = self.images.tiles(q.cube, self.composites)
            tiles += t
            grids += g
        last = np.asarray([b * L + len(p) - 1 for b, p in enumerate(prompts)], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(att),
            "mm_token_type_ids": torch.from_numpy(
                (ids == self.renderer.image_pad).astype(np.int32)
            ),
            "pixel_tiles": torch.from_numpy(np.concatenate(tiles)),
            "image_grid_thw": torch.from_numpy(np.asarray(grids, dtype=np.int64)),
            "last_pos": torch.from_numpy(last),
        }

    def run(self, lm: Any, questions: list[EvalQuestion], device: Any) -> list[dict[str, Any]]:
        """``lm``: a ``LabelPositionLM`` (unwrapped from DDP)."""
        import torch

        records = []
        dtype = lm.base.lm_head.weight.dtype
        with (
            torch.no_grad(),
            torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"),
        ):
            for i in range(0, len(questions), self.batch_size):
                qs = questions[i : i + self.batch_size]
                batch = {k: v.to(device, non_blocking=True) for k, v in self.batch(qs).items()}
                h = lm.hidden_states(batch, dtype)
                h = h.reshape(-1, h.shape[-1]).index_select(0, batch["last_pos"])
                logits = lm.base.lm_head(h).float()
                for q, row in zip(qs, logits, strict=True):
                    cids = torch.tensor([self.cand_id(c) for c in q.candidates], device=device)
                    probs = torch.softmax(row[cids], -1).cpu().numpy()
                    k = int(probs.argmax())
                    records.append(
                        {
                            "qid": q.qid,
                            "patch_id": q.patch_id,
                            "type": q.type,
                            "category": q.category,
                            "answer": q.answer,
                            "pred": q.candidates[k],
                            "confidence": float(probs[k]),
                            "correct": q.candidates[k] == q.answer,
                            "probs": {
                                c: float(p) for c, p in zip(q.candidates, probs, strict=True)
                            },
                            "top_vocab": int(row.argmax()),
                        }
                    )
        return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Paper-style breakdown: per type, per category accuracy and overall (OA)."""
    out: dict[str, Any] = {"n": len(records)}
    for typ, cats in (("binary", PAPER_BINARY), ("mcq", PAPER_MCQ)):
        rs = [r for r in records if r["type"] == typ]
        if not rs:
            continue
        per = {}
        for c in cats:
            sub = [r["correct"] for r in rs if r["category"] == c]
            if sub:
                per[ABBREV[c]] = {"acc": 100 * float(np.mean(sub)), "n": len(sub)}
        out[typ] = {"OA": 100 * float(np.mean([r["correct"] for r in rs])), "n": len(rs), **per}
    oas = [out[t]["OA"] for t in ("binary", "mcq") if t in out]
    out["mean_OA"] = float(np.mean(oas)) if oas else 0.0
    return out


def calibration(records: list[dict[str, Any]], n_bins: int = 10) -> dict[str, Any]:
    """Reliability bins + expected calibration error, per answer type."""
    res = {}
    for typ in ("binary", "mcq"):
        rs = [r for r in records if r["type"] == typ]
        if not rs:
            continue
        lo = 1.0 / len(rs[0]["probs"])  # chance level confidence
        edges = np.linspace(lo, 1.0, n_bins + 1)
        conf = np.array([r["confidence"] for r in rs])
        corr = np.array([r["correct"] for r in rs], dtype=float)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
        bins, ece = [], 0.0
        for b in range(n_bins):
            m = idx == b
            if m.any():
                bins.append(
                    {
                        "lo": float(edges[b]),
                        "hi": float(edges[b + 1]),
                        "n": int(m.sum()),
                        "conf": float(conf[m].mean()),
                        "acc": float(corr[m].mean()),
                    }
                )
                ece += m.mean() * abs(conf[m].mean() - corr[m].mean())
        res[typ] = {"ece": float(ece), "bins": bins}
    return res
