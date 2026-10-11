"""M1 dataset analysis of BigEarthNet.txt (parquet only, CPU).

    uv run --no-sync python scripts/analyze_dataset.py \
        --parquet datasets/bigearthnet-txt/BigEarthNet.txt.parquet --out reports

Writes ``dataset_analysis.md`` (human-readable, with a "contradictions" section comparing
against the Phase 2.1 brief) and ``dataset_analysis.json`` (numbers reused later).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benvqa.data.mcq import parse_mcq
from benvqa.prompting import IMAGE_TAGS, SYSTEM_PROMPT, TOKENIZER_ID

BRIEF_TYPES = {
    "binary": 3_625_160,
    "mcq": 3_259_184,
    "captioning": 463_932,
    "bounding box": 2_205_686,
}
BRIEF_SPLITS = {"train": 4_674_281, "validation": 2_454_690, "test": 2_409_962, "bench": 15_029}
BRIEF_IMAGES = {"train": 229_114, "validation": 118_095, "test": 116_835}
SPLITS = ["train", "validation", "test", "bench"]
VQA_CATS = [
    "presence",
    "area",
    "count",
    "adjacency",
    "relative pos",
    "country",
    "season",
    "climate zone",
]
TOKENS_PER_IMAGE = 64  # 256x256 input, patch 16, merge 2


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return [*out, ""]


def load(parquet: Path) -> pd.DataFrame:
    cols = [
        "patch_id",
        "s1_name",
        "input",
        "output",
        "type",
        "category",
        "split",
        "country",
        "season",
        "climate_zone",
    ]
    df = pq.read_table(parquet, columns=cols).to_pandas()
    # Caption rows carry the literal string "None" as category.
    cap = (df["type"] == "captioning") | df["category"].isna()
    df.loc[cap, "category"] = "captioning"
    return df


def counts_section(df: pd.DataFrame, res: dict, flags: list[str]) -> list[str]:
    md = ["## 1. Row and image counts", ""]
    by_type = df["type"].value_counts().to_dict()
    res["rows_by_type"] = by_type
    md += _table(
        ["type", "rows", "brief", "match"],
        [[t, by_type.get(t, 0), b, by_type.get(t, 0) == b] for t, b in BRIEF_TYPES.items()],
    )
    cat = (
        df[df.type.isin(["binary", "mcq"])]
        .groupby(["category", "type"])
        .size()
        .unstack(fill_value=0)
    )
    cat["total"] = cat.sum(axis=1)
    res["rows_by_category"] = {
        k: {kk: int(vv) for kk, vv in v.items()} for k, v in cat.T.to_dict().items()
    }
    md += _table(
        ["category", "binary", "mcq", "total"],
        [[c, int(r["binary"]), int(r["mcq"]), int(r["total"])] for c, r in cat.iterrows()],
    )
    by_split = df["split"].value_counts().to_dict()
    images = df.groupby("split")["patch_id"].nunique().to_dict()
    res["rows_by_split"], res["images_by_split"] = by_split, images
    md += _table(
        ["split", "rows", "brief rows", "images", "brief images"],
        [
            [s, by_split[s], BRIEF_SPLITS[s], images[s], BRIEF_IMAGES.get(s, "1,082")]
            for s in SPLITS
        ],
    )
    if images["test"] != BRIEF_IMAGES["test"]:
        flags.append(
            f"Test has {images['test']:,} images, not {BRIEF_IMAGES['test']:,}. The brief's "
            f"figure equals test + bench ({images['test'] + images['bench']:,}): the 1,082 "
            "bench images are a separate set, not part of test."
        )
    sets = {s: set(df.patch_id[df.split == s].unique()) for s in SPLITS}
    overlaps = {
        f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1 :]
    }
    s1 = {s: set(df.s1_name[df.split == s].unique()) for s in SPLITS}
    overlaps["s1:train&(test|bench)"] = len(s1["train"] & (s1["test"] | s1["bench"]))
    res["split_overlaps"] = overlaps
    md += ["Image overlap between splits (S2 `patch_id`; S1 names for train vs test/bench):", ""]
    md += _table(["pair", "shared images"], [[k, v] for k, v in overlaps.items()])
    n_s1 = df.groupby("patch_id")["s1_name"].nunique()
    md += [f"Patches with more than one `s1_name`: {(n_s1 > 1).sum()}.", ""]
    return md


def structure_section(df: pd.DataFrame, res: dict, flags: list[str]) -> list[str]:
    md = ["## 2. Per-image structure (binary / MCQ / caption)", ""]
    d = df[df.type.isin(["binary", "mcq", "captioning"])]
    key = d.type.str[:3] + ":" + d.category + np.where(d.type == "binary", ":" + d.output, "")
    per = d.assign(key=key).groupby(["split", "patch_id", "key"]).size()
    res["structure"] = {}
    for split in SPLITS:
        c = per.loc[split].unstack(fill_value=0)
        n = len(c)
        rows, stats = [], {}
        for k in sorted(c.columns):
            vc = c[k].value_counts().sort_index()
            stats[k] = {
                "has": float((c[k] > 0).mean()),
                "mean": float(c[k].mean()),
                "dist": {int(a): int(b) for a, b in vc.items()},
            }
            dist = ", ".join(f"{a}:{b}" for a, b in list(vc.items())[:6])
            rows.append([k, _pct(stats[k]["has"]), f"{stats[k]['mean']:.2f}", dist])
        res["structure"][split] = {"images": n, "keys": stats}
        md += [f"### {split} ({n:,} images)", ""]
        md += _table(["item", "images with it", "mean per image", "count:images"], rows)
    tr = res["structure"]["train"]["keys"]
    adj_mcq, adj_no = tr["mcq:adjacency"]["has"], tr["bin:adjacency:no"]["has"]
    rp = tr["mcq:relative pos"]["has"]
    flags.append(
        f"Adjacency: every train image has the binary `no` question ({_pct(adj_no)}), but the "
        f"binary `yes` and the MCQ exist only when the scene has a touching pair "
        f"({_pct(adj_mcq)} of train images, not ~88%). Picking among the *available* items "
        "would make adjacency binaries ~2:1 `no`. The conversation builder therefore treats "
        "the group pick probabilities as absolute, and an unavailable pick skips the group."
    )
    flags.append(f"Relative position MCQ exists for {_pct(rp)} of train images (brief: ~21%).")
    bench = res["structure"]["bench"]["keys"]
    flags.append(
        "Bench is structured differently from train: several questions per category per "
        f"image (e.g. up to {max(bench['bin:adjacency:yes']['dist'])} adjacency `yes` "
        "binaries), relative-position MCQs come in pairs (both directions), and season / "
        f"climate / country MCQs exist for only {_pct(bench['mcq:season']['has'])} / "
        f"{_pct(bench['mcq:climate zone']['has'])} / {_pct(bench['mcq:country']['has'])} of "
        "bench images. Fine for single-turn eval. In train every image has exactly one of each."
    )
    return md


def answers_section(df: pd.DataFrame, res: dict, flags: list[str]) -> list[str]:
    md = ["## 3. Answer balance: train vs bench", ""]
    res["answers"] = {}
    rows = []
    for split in ["train", "bench"]:
        d = df[(df.split == split) & df.type.isin(["binary", "mcq"])]
        tab = (
            d.groupby(["type", "category"])["output"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        res["answers"][split] = {
            f"{t}:{c}": {k: float(v) for k, v in r.items() if v > 0} for (t, c), r in tab.iterrows()
        }
    for k in sorted(res["answers"]["train"]):
        tr, be = res["answers"]["train"][k], res["answers"]["bench"].get(k, {})
        letters = ["yes", "no"] if k.startswith("binary") else ["a", "b", "c", "d"]
        rows.append(
            [
                k,
                " / ".join(_pct(tr.get(x, 0)) for x in letters),
                " / ".join(_pct(be.get(x, 0)) for x in letters),
            ]
        )
    md += _table(["type:category", "train (yes/no or a/b/c/d)", "bench"], rows)
    other = df[(df.type == "binary") & ~df.output.isin(["yes", "no"])]
    other_m = df[(df.type == "mcq") & ~df.output.isin(list("abcd"))]
    md += [
        f"Binary outputs other than yes/no: {len(other)}; MCQ outputs other than a-d: "
        f"{len(other_m)}.",
        "",
    ]
    res["answer_format_violations"] = {"binary": len(other), "mcq": len(other_m)}

    md += ["### Metadata priors (fraction of images)", ""]
    img = df.drop_duplicates("patch_id")
    for col in ["country", "season", "climate_zone"]:
        tr = img[img.split == "train"][col].value_counts(normalize=True)
        be = img[img.split == "bench"][col].value_counts(normalize=True)
        res[f"prior_{col}"] = {"train": tr.to_dict(), "bench": be.to_dict()}
        md += _table(
            [col, "train", "bench"], [[k, _pct(v), _pct(be.get(k, 0))] for k, v in tr.items()]
        )
    return md


def mcq_section(df: pd.DataFrame, res: dict, flags: list[str]) -> list[str]:
    md = ["## 4. MCQ option parsing", ""]
    m = df[df.type == "mcq"]
    fails: Counter[str] = Counter()
    examples: list[str] = []
    n_opts: Counter[int] = Counter()
    dup = 0
    for text, cat in zip(m["input"].to_numpy(), m["category"].to_numpy(), strict=True):
        parsed = parse_mcq(text)
        if parsed is None:
            fails[cat] += 1
            if len(examples) < 5:
                examples.append(text)
            continue
        n_opts[len(parsed.options)] += 1
        dup += len(set(parsed.options)) != len(parsed.options)
    total = len(m)
    rate = 1 - sum(fails.values()) / total
    res["mcq_parse"] = {
        "rows": total,
        "success_rate": rate,
        "fail_by_category": dict(fails),
        "n_options": dict(n_opts),
        "duplicate_options": dup,
    }
    md += [
        f"Parsed and round-tripped exactly: {total - sum(fails.values()):,} / {total:,} "
        f"({100 * rate:.4f}%). Options per question: {dict(n_opts)}. Questions with "
        f"duplicate option text (not shuffled): {dup}.",
        "",
    ]
    if fails:
        md += ["Failures by category: " + ", ".join(f"{k} {v}" for k, v in fails.items()), ""]
        md += ["Examples:", "", *[f"- `{e}`" for e in examples], ""]
    commas = m["input"].str.count(",") > 3
    md += [
        f"MCQs whose option text contains commas: {int(commas.sum()):,} ({_pct(commas.mean())}).",
        "",
    ]
    return md


def malformed_section(df: pd.DataFrame, res: dict, flags: list[str]) -> list[str]:
    md = ["## 5. Malformed / suspicious rows", ""]
    bb = df[df.type == "bounding box"]
    double = bb["input"].str.count(r"<ref>|<point>") > 1
    box = re.compile(r"^\[\s*-?[\d.]+\s+-?[\d.]+\s*,\s*-?[\d.]+\s+-?[\d.]+\s*\]$")
    bad_box = ~bb["output"].map(lambda s: bool(box.match(s)))
    empty = (df["input"].str.strip() == "") | (df["output"].str.strip() == "")
    dup = df.duplicated(["patch_id", "type", "input"]).sum()
    area = df[(df.type == "binary") & (df.category == "area")]
    trivial = area["input"].str.contains(
        r"(?:at least|greater than or equal to|more than or equal to) 0(?:%| m2)\b", regex=True
    )
    res["malformed"] = {
        "bbox_multiple_instructions": int(double.sum()),
        "bbox_unparseable_output": int(bad_box.sum()),
        "empty_input_or_output": int(empty.sum()),
        "duplicate_patch_type_input": int(dup),
        "area_binary_trivial_zero_threshold": int(trivial.sum()),
        "area_binary_trivial_answers": area[trivial]["output"].value_counts().to_dict(),
    }
    md += _table(["check", "rows"], [[k, v] for k, v in res["malformed"].items()])
    md += ["Examples of bbox inputs with two instructions:", ""]
    md += [f"- `{s[:220]}`" for s in bb[double]["input"].head(3)] + [""]
    if trivial.sum():
        flags.append(
            f"{int(trivial.sum()):,} area binaries use a zero threshold ('at least 0 m2', "
            "'greater than or equal to 0%'). They are always `yes` and only test class "
            "presence. Kept: they are correct, and presence ordering already covers the leak."
        )
    return md


def tokens_section(df: pd.DataFrame, res: dict, flags: list[str], tok: Any, n: int) -> list[str]:
    md = ["## 6. Token lengths (Qwen3.5 tokenizer)", ""]
    rng = np.random.default_rng(0)
    rows, lens = [], {}
    no_special = {"add_special_tokens": False}
    d = df[df.split == "train"]
    for (t, c), g in d[d.type.isin(["binary", "mcq", "captioning"])].groupby(["type", "category"]):
        take = g.iloc[rng.choice(len(g), size=min(n, len(g)), replace=False)]
        q = np.array([len(x) for x in tok(list(take["input"]), **no_special)["input_ids"]])
        a = np.array([len(x) for x in tok(list(take["output"]), **no_special)["input_ids"]])
        lens[f"{t}:{c}"] = {
            "q_mean": float(q.mean()),
            "a_mean": float(a.mean()),
            "q_p95": float(np.percentile(q, 95)),
            "a_p95": float(np.percentile(a, 95)),
            "a_max": int(a.max()),
        }
        rows.append(
            [
                f"{t}:{c}",
                f"{q.mean():.1f}",
                int(np.percentile(q, 95)),
                f"{a.mean():.1f}",
                int(np.percentile(a, 95)),
                int(a.max()),
            ]
        )
    res["token_lengths"] = lens
    md += _table(["type:category", "question mean", "q p95", "answer mean", "a p95", "a max"], rows)

    single = {w: tok(w, add_special_tokens=False)["input_ids"] for w in ["yes", "no", *"abcd"]}
    res["answer_token_ids"] = single
    ok = all(len(v) == 1 for v in single.values())
    md += [
        f"Answer words as tokens: {single}. All single tokens: **{ok}** (checked in context "
        "after the assistant prefix in the M3 tests).",
        "",
    ]
    if not ok:
        flags.append("Some answer words are not single tokens; score-based eval must handle it.")
    return md


def projection_section(res: dict, tok: Any) -> list[str]:
    """Expected supervised answers and tokens per sample: multi-turn conversation vs single-turn."""

    def n(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    md = ["## 7. Projected tokens per sample", ""]
    sys_t = n(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
    img_t = {
        k: n(f"{tag}<|vision_start|><|vision_end|>\n") + TOKENS_PER_IMAGE
        for k, tag in IMAGE_TAGS.items()
    }
    user_wrap = n("<|im_start|>user\n<|im_end|>\n")
    asst_wrap = n("<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n")
    L = res["token_lengths"]
    keys = res["structure"]["train"]["keys"]

    def turn(k: str) -> float:
        return user_wrap + asst_wrap + L[k]["q_mean"] + L[k]["a_mean"]

    # Expected selection under the default config: group pick 0.25/0.25/0.5 (unavailable ->
    # skip), p_drop_presence 0.3, relative_pos always, geo = one of country/climate.
    p = {"bin": 0.5, "mcq": 0.5}
    exp_turns, exp_tok = 0.0, 0.0
    for cat in ["presence", "area", "count", "adjacency"]:
        keep = 0.7 if cat == "presence" else 1.0
        for kind, prob in (("bin", p["bin"]), ("mcq", p["mcq"])):
            key = f"{kind}:{cat}" if kind == "mcq" else f"bin:{cat}:yes"
            if kind == "bin":
                avail = 0.5 * keys[f"bin:{cat}:yes"]["has"] + 0.5 * keys[f"bin:{cat}:no"]["has"]
            else:
                avail = keys[key]["has"]
            tk = f"{'binary' if kind == 'bin' else 'mcq'}:{cat}"
            exp_turns += keep * prob * avail
            exp_tok += keep * prob * avail * turn(tk)
    for cat, share in (
        ("relative pos", keys["mcq:relative pos"]["has"]),
        ("season", 1.0),
        ("country", 0.5),
        ("climate zone", 0.5),
    ):
        exp_turns += share
        exp_tok += share * turn(f"mcq:{cat}")
    cap = 0.2 * turn("captioning:captioning")
    images3 = sum(img_t[k] for k in ("rgb", "nir_swir", "sar"))
    conv = sys_t + images3 + exp_tok + cap
    mean_turn = exp_tok / exp_turns
    single = sys_t + images3 + mean_turn
    res["projection"] = {
        "system": sys_t,
        "images_3": images3,
        "qa_turns": exp_turns,
        "qa_tokens": exp_tok,
        "caption_tokens": cap,
        "conversation": conv,
        "single_turn": single,
        "answers_per_1k_tokens_conv": 1000 * exp_turns / conv,
        "answers_per_1k_tokens_single": 1000 / single,
    }
    md += _table(
        ["", "tokens"],
        [
            ["system prompt", sys_t],
            ["3 images (64 visual + tag/markers each)", images3],
            ["expected QA turns per conversation", f"{exp_turns:.2f}"],
            ["QA turn tokens (sum, expected)", f"{exp_tok:.0f}"],
            ["caption turn (p=0.2, expected)", f"{cap:.0f}"],
            ["**conversation total**", f"**{conv:.0f}**"],
            ["single-turn sample", f"{single:.0f}"],
            [
                "answers per 1k tokens: conversation / single",
                f"{1000 * exp_turns / conv:.2f} / {1000 / single:.2f}",
            ],
        ],
    )
    gain = (exp_turns / conv) / (1 / single)
    md += [
        f"Multi-turn packing gives **{gain:.1f}x** more supervised answers per token. "
        "Per GPU-second the gain is larger, because the vision encoder runs once per "
        "image set rather than once per answer.",
        "",
    ]
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default = Path("datasets/bigearthnet-txt/BigEarthNet.txt.parquet")
    ap.add_argument("--parquet", type=Path, default=default)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--token-samples", type=int, default=20_000)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    df = load(args.parquet)
    res: dict[str, Any] = {"parquet": str(args.parquet), "rows": len(df)}
    flags: list[str] = []
    body: list[str] = []
    for fn in (counts_section, structure_section, answers_section, mcq_section, malformed_section):
        body += fn(df, res, flags)
        print(f"{fn.__name__} done ({time.time() - t0:.0f}s)")
    body += tokens_section(df, res, flags, tok, args.token_samples)
    body += projection_section(res, tok)
    md = [
        "# BigEarthNet.txt dataset analysis (M1)",
        "",
        f"Source: `{args.parquet}` ({len(df):,} rows). Generated by `scripts/analyze_dataset.py` "
        f"in {time.time() - t0:.0f} s.",
        "",
        "## Contradictions with / additions to the brief",
        "",
        *[f"- {f}" for f in flags],
        "",
        *body,
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_analysis.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    (args.out / "dataset_analysis.json").write_text(
        json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.out / 'dataset_analysis.md'}")


if __name__ == "__main__":
    main()
