"""Parse, shuffle and re-render BigEarthNet.txt multiple-choice questions.

Options are inline: ``"<question> a) Lithuania, b) Belgium, c) Austria, d) Luxembourg"``.
Option text can itself contain commas (``"Cold, no dry season, warm summer"``), so we never
split on commas. Instead we locate the markers ``a) ``, ``, b) ``, ``, c) ``, ``, d) ``
from the right and accept a parse only if rendering it reproduces the input exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LETTERS = "abcdefgh"


@dataclass(frozen=True)
class MCQ:
    prefix: str
    """Everything before ``a) ``, including its trailing separator."""
    options: tuple[str, ...]

    def render(self) -> str:
        return self.prefix + ", ".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(self.options))

    @property
    def letters(self) -> str:
        return LETTERS[: len(self.options)]


def parse_mcq(text: str, max_options: int = 4) -> MCQ | None:
    """Return the parsed question, or ``None`` if it does not round-trip exactly."""
    for n in range(max_options, 1, -1):
        cuts: list[int] = []
        end = len(text)
        ok = True
        for i in range(n - 1, 0, -1):
            pos = text.rfind(f", {LETTERS[i]}) ", 0, end)
            if pos < 0:
                ok = False
                break
            cuts.append(pos)
            end = pos
        if not ok:
            continue
        a_pos = text.rfind("a) ", 0, end)
        if a_pos < 0 or (a_pos > 0 and not text[a_pos - 1].isspace()):
            continue
        cuts.reverse()  # positions of ", b) ", ", c) ", ...
        starts = [a_pos + 3] + [c + 5 for c in cuts]
        ends = [*cuts, len(text)]
        options = tuple(text[s:e] for s, e in zip(starts, ends, strict=True))
        mcq = MCQ(prefix=text[:a_pos], options=options)
        if all(options) and mcq.render() == text:
            return mcq
    return None


def shuffle_mcq(text: str, answer: str, rng: np.random.Generator) -> tuple[str, str, bool]:
    """Permute the options and remap the answer letter.

    Returns ``(text, answer, shuffled)``. Falls back to the original when the question does
    not parse, the answer is not one of the offered letters, or option texts repeat (the
    remapped answer would be ambiguous).
    """
    mcq = parse_mcq(text)
    if mcq is None or answer not in mcq.letters or len(set(mcq.options)) != len(mcq.options):
        return text, answer, False
    perm = rng.permutation(len(mcq.options))
    correct = mcq.letters.index(answer)
    options = tuple(mcq.options[j] for j in perm)
    new_answer = LETTERS[int(np.flatnonzero(perm == correct)[0])]
    return MCQ(mcq.prefix, options).render(), new_answer, True
