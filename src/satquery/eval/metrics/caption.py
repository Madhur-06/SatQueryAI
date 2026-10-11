"""Captioning metrics via pycocoevalcap ([eval] extra): BLEU-4, METEOR, ROUGE-L, CIDEr.

Tokenisation is a simple lowercase word split (not the Java PTB tokenizer), so BLEU and
CIDEr need no Java; METEOR runs the bundled Java jar and is reported as ``None`` if Java
is unavailable. Scores are therefore comparable across our own runs; small differences
from numbers computed with the PTB tokenizer are expected.
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def caption_scores(
    predictions: dict[str, str], references: dict[str, list[str]], meteor: bool = True
) -> dict[str, Any]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise ImportError("Captioning metrics need the [eval] extra (pycocoevalcap).") from exc

    ids = sorted(set(predictions) & set(references))
    if not ids:
        return {"n": 0, "bleu4": None, "meteor": None, "rouge_l": None, "cider": None}
    res = {i: [tokenize(predictions[i]) or "empty"] for i in ids}
    gts = {i: [tokenize(r) for r in references[i]] for i in ids}
    with contextlib.redirect_stdout(io.StringIO()):  # Bleu prints debug output
        bleu, _ = Bleu(4).compute_score(gts, res)
        cider, _ = Cider().compute_score(gts, res)
        rouge, _ = Rouge().compute_score(gts, res)
    out: dict[str, Any] = {
        "n": len(ids),
        "bleu4": float(bleu[3]),
        "rouge_l": float(rouge),
        "cider": float(cider),
    }
    out["meteor"] = None
    if meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor

            out["meteor"] = float(Meteor().compute_score(gts, res)[0])
        except (OSError, FileNotFoundError, RuntimeError) as exc:  # Java missing / crashed
            out["meteor_error"] = f"{type(exc).__name__}: {exc}"
    return out
