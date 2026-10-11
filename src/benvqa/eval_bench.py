"""Stage 4: BigEarthNet.txt bench evaluation (score-based VQA + greedy captioning).

    # zero-shot baselines (M6), fine-tuned (M8); one GPU or torchrun --nproc_per_node 2
    python -m benvqa.eval_bench --config configs/kaggle_t4x2.yaml --runs a,b
    python -m benvqa.eval_bench --config configs/kaggle_t4x2.yaml --runs c,d,e --captions 200

Runs (Phase 2.1 brief section 9):
  a  zero-shot Qwen3.5-4B, RGB only       b  zero-shot, rgb + nir_swir + sar
  c  fine-tuned, all three composites     d  fine-tuned, RGB only     e  fine-tuned, SAR only
(d) vs (c) is the spectral/SAR ablation. "Zero-shot" = the same model with a freshly
initialised LoRA, whose B matrices are zero, so it is exactly the base model.

Bench inputs are used verbatim (no option shuffling). Every run writes
``reports/bench_<run>.json`` and ``.records.jsonl.gz``. ``reports/bench_results.md`` and
the reliability diagrams are rebuilt from every run file present.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benvqa.config import Config, load_config
from benvqa.data.render import Renderer
from benvqa.data.stream import ImageMaker, default_cache_dir, load_quant_spec
from benvqa.data.vision import to_pixel_values
from benvqa.evaluation import (
    ABBREV,
    PAPER_BINARY,
    PAPER_MCQ,
    EvalQuestion,
    ScoreEvaluator,
    calibration,
    load_split,
    summarize,
    vqa_questions,
)
from benvqa.hub_sync import make_mirror
from benvqa.logging_utils import setup_logging
from benvqa.model.load import load_adapter_weights, load_model
from benvqa.model.loss import LabelPositionLM
from benvqa.train import Dist, get_tokenizer

log = logging.getLogger("benvqa.eval_bench")
ALL = ("rgb", "nir_swir", "sar")
RUNS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "a": ("zero-shot", ("rgb",), "{model} zero-shot, RGB"),
    "b": ("zero-shot", ALL, "{model} zero-shot, RGB + NIR-SWIR + SAR"),
    "c": ("fine-tuned", ALL, "{model} + our LoRA, RGB + NIR-SWIR + SAR"),
    "d": ("fine-tuned", ("rgb",), "{model} + our LoRA, RGB only"),
    "e": ("fine-tuned", ("sar",), "{model} + our LoRA, SAR only"),
}
PAPER = [  # BigEarthNet.txt paper (arXiv 2603.29630), bench split, overall accuracy
    ("GPT-5.2 (zero-shot)", 60.39, 34.93),
    ("Qwen3-VL-8B (zero-shot, best open model)", 61.96, 37.55),
    ("RS-InternVL (fine-tuned, 4xH200 ~2 days)", 73.29, 51.49),
]


# ---------------------------------------------------------------------------- adapter
def fetch_adapter(spec: str, cache: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """``hub:<user>/<repo>[:<path>]`` (default path ``ckpt/best``) or a local directory."""
    from safetensors.torch import load_file

    if spec.startswith(("hub:", "local:")):
        if spec.startswith("hub:"):
            repo, _, sub = spec[4:].partition(":")
            mirror = make_mirror(f"hub:{repo}", repo_type="model")
        else:
            mirror, sub = make_mirror(spec), ""
        sub = (sub or "ckpt/best").strip("/")
        for name in ("adapter_model.safetensors", "adapter_config.json", "best_metrics.json"):
            try:
                mirror.download(f"{sub}/{name}", cache)
            except FileNotFoundError:
                if name != "best_metrics.json":
                    raise
        folder = cache / sub
    else:
        folder = Path(spec)
    meta = {"source": spec}
    if (folder / "best_metrics.json").exists():
        meta["best_metrics"] = json.loads((folder / "best_metrics.json").read_text())
    elif (folder / "trainer_state.json").exists():
        meta["step"] = json.loads((folder / "trainer_state.json").read_text()).get("step")
    return load_file(str(folder / "adapter_model.safetensors")), meta


# --------------------------------------------------------------------------- captions
def caption_questions(rows: list[dict[str, Any]], n: int) -> list[EvalQuestion]:
    out = [
        EvalQuestion(
            it.id, r["patch_id"], r["cube"], it.input, it.output, "captioning", "captioning", ()
        )
        for r in rows
        for it in r["qa"]
        if it.type == "captioning"
    ]
    return sorted(out, key=lambda q: q.qid)[:n]


@torch.no_grad()
def generate_captions(
    loaded: Any,
    tok: Any,
    renderer: Renderer,
    images: ImageMaker,
    comps: tuple[str, ...],
    qs: list[EvalQuestion],
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    dtype = loaded.base.lm_head.weight.dtype
    out = []
    for q in qs:
        ids = torch.tensor([renderer.prompt_ids(q.question, comps)], device=device)
        tiles, grids = images.tiles(q.cube, comps)
        tiles_t = torch.from_numpy(np.concatenate(tiles)).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            gen = loaded.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                mm_token_type_ids=(ids == renderer.image_pad).int(),
                pixel_values=to_pixel_values(tiles_t, dtype),
                image_grid_thw=torch.tensor(grids, device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = tok.decode(gen[0, ids.shape[1] :].tolist(), skip_special_tokens=True).strip()
        out.append(
            {"qid": q.qid, "patch_id": q.patch_id, "prediction": text, "reference": q.answer}
        )
    return out


# ------------------------------------------------------------------------------ report
def _git() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def reliability_svg(cal: dict[str, Any], title: str) -> str:
    """Reliability diagram (confidence vs accuracy per bin), light/dark aware, hover titles."""
    w, h, m = 420, 380, {"l": 52, "r": 24, "t": 62, "b": 46}
    pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]

    def x(v: float) -> float:
        return m["l"] + v * pw

    def y(v: float) -> float:
        return m["t"] + (1 - v) * ph

    series = [("binary", "Binary", "s1"), ("mcq", "MCQ", "s2")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img" aria-label="{title}">',
        "<style>svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;"
        "--grid:#e1e0d9;--s1:#2a78d6;--s2:#eb6834;font-family:system-ui,sans-serif}"
        "@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#ffffff;"
        "--ink2:#c3c2b7;--grid:#2c2c2a;--s1:#3987e5;--s2:#d95926}}"
        "text{fill:var(--ink2);font-size:11px}.t{fill:var(--ink);font-size:13px;"
        "font-weight:600}</style>",
        f'<rect width="{w}" height="{h}" fill="var(--surface)"/>',
        f'<text class="t" x="{m["l"]}" y="20">{title}</text>',
    ]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts += [
            f'<line x1="{x(0)}" x2="{x(1)}" y1="{y(v)}" y2="{y(v)}" stroke="var(--grid)" '
            'stroke-width="1"/>',
            f'<text x="{m["l"] - 8}" y="{y(v) + 4}" text-anchor="end">{v:.2f}</text>',
            f'<text x="{x(v)}" y="{h - m["b"] + 16}" text-anchor="middle">{v:.2f}</text>',
        ]
    parts += [
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="var(--muted)" '
        'stroke-width="1" stroke-dasharray="4 4"/>',
        f'<text x="{x(0.06)}" y="{y(0.02)}" transform="rotate(-45 {x(0.06)} {y(0.02)})">'
        "perfect calibration</text>",
        f'<text x="{m["l"] + pw / 2}" y="{h - 8}" text-anchor="middle">confidence (softmax over'
        " offered answers)</text>",
        f'<text x="14" y="{m["t"] + ph / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {m["t"] + ph / 2})">accuracy</text>',
    ]
    lx = m["l"]
    for key, label, var in series:  # legend row under the title (identity never color-only)
        if key in cal:
            text = f"{label} (ECE {cal[key]['ece']:.3f})"
            parts += [
                f'<line x1="{lx}" x2="{lx + 14}" y1="38" y2="38" stroke="var(--{var})" '
                'stroke-width="2"/>',
                f'<circle cx="{lx + 7}" cy="38" r="3" fill="var(--{var})"/>',
                f'<text x="{lx + 20}" y="42">{text}</text>',
            ]
            lx += 26 + 6.2 * len(text)
    for key, label, var in series:
        if key not in cal or not cal[key]["bins"]:
            continue
        pts = [(x(b["conf"]), y(b["acc"]), b) for b in cal[key]["bins"]]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py, _) in enumerate(pts)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="var(--{var})" stroke-width="2"/>')
        for px, py, b in pts:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="var(--{var})" '
                f'stroke="var(--surface)" stroke-width="2"><title>{label}: confidence '
                f"{b['conf']:.2f}, accuracy {b['acc']:.2f}, n={b['n']}</title></circle>"
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_results_md(out_dir: Path) -> Path:
    runs = []
    for f in sorted(out_dir.glob("bench_?.json")):
        runs.append(json.loads(f.read_text()))
    lines = [
        "# BigEarthNet.txt bench results",
        "",
        "Score-based single-turn evaluation on the manually verified **bench** split "
        "(answers read from the logits of the offered candidates; inputs verbatim). "
        "Generated by `benvqa.eval_bench`.",
        "",
        "## VQA accuracy (%)",
        "",
        "| Model | "
        + " | ".join(f"Bin {ABBREV[c]}" for c in PAPER_BINARY)
        + " | **Bin OA** | "
        + " | ".join(f"MCQ {ABBREV[c]}" for c in PAPER_MCQ)
        + " | **MCQ OA** |",
        "|---|" + "---|" * (len(PAPER_BINARY) + len(PAPER_MCQ) + 2),
    ]
    blank_b, blank_m = ["-"] * len(PAPER_BINARY), ["-"] * len(PAPER_MCQ)
    for name, b, m in PAPER:
        lines.append(
            f"| {name} (paper) | "
            + " | ".join(blank_b)
            + f" | {b:.2f} | "
            + " | ".join(blank_m)
            + f" | {m:.2f} |"
        )
    for r in runs:
        s = r["summary"]

        def cell(t: str, c: str, s: dict = s) -> str:
            return _f(s.get(t, {}).get(ABBREV[c], {}).get("acc"))

        lines.append(
            f"| ({r['run']}) {r['label']} | "
            + " | ".join(cell("binary", c) for c in PAPER_BINARY)
            + f" | **{_f(s.get('binary', {}).get('OA'))}** | "
            + " | ".join(cell("mcq", c) for c in PAPER_MCQ)
            + f" | **{_f(s.get('mcq', {}).get('OA'))}** |"
        )
    lines += [
        "",
        "Paper per-category numbers are not reproduced here (only the overall "
        "accuracies quoted in the brief); `-` = not reported.",
        "",
    ]
    caps = [r for r in runs if r.get("captions")]
    if caps:
        lines += [
            "## Captioning (greedy)",
            "",
            "| Run | n | BLEU-4 | METEOR | ROUGE-L | CIDEr |",
            "|---|---|---|---|---|---|",
        ]
        for r in caps:
            c = r["captions"]
            lines.append(
                f"| ({r['run']}) {r['label']} | {c.get('n')} | {_f(c.get('bleu4'))} | "
                f"{_f(c.get('meteor'))} | {_f(c.get('rouge_l'))} | {_f(c.get('cider'))} |"
            )
        lines += [
            "",
            "RS-InternVL (paper): BLEU-4 34.04. Our scores use a simple lower-case "
            "word tokenizer (`satquery.eval.metrics.caption`), not the PTB tokenizer.",
            "",
        ]
    lines += [
        "## Calibration of the confidence score",
        "",
        "| Run | Binary ECE | MCQ ECE | Diagram |",
        "|---|---|---|---|",
    ]
    for r in runs:
        cal = r["calibration"]
        lines.append(
            f"| ({r['run']}) | {_f(cal.get('binary', {}).get('ece'))} | "
            f"{_f(cal.get('mcq', {}).get('ece'))} | ![](reliability_{r['run']}.svg) |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        "| Run | adapter | n questions | git | seconds |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| ({r['run']}) | {(r.get('adapter') or {}).get('source', 'none (zero-shot)')} | "
            f"{r['summary']['n']} | {r.get('git')} | {r.get('seconds')} |"
        )
    path = out_dir / "bench_results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--runs", default="a,b,c,d,e", help="comma list of a..e")
    ap.add_argument(
        "--adapter",
        default=None,
        help="hub:<user>/<repo>[:ckpt/best] or a local folder; default "
        "checkpoint.hub_repo + ckpt/best",
    )
    ap.add_argument("--split", default="bench")
    ap.add_argument("--limit", type=int, default=None, help="first N questions (debugging)")
    ap.add_argument("--captions", type=int, default=0, help="caption N bench images (greedy)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    d = Dist()
    setup_logging(d.rank)
    cfg: Config = load_config(args.config, args.set)
    if cfg.model.backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers is used)
    cfg.data.cache_dir = default_cache_dir(cfg.data.cache_dir)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    runs.sort(key=lambda r: RUNS[r][0] != "zero-shot")  # zero-shot before loading the adapter

    spec = cfg.data.hub_dataset
    mirror = make_mirror(
        spec, **({} if spec.startswith("local:") else {"revision": cfg.data.revision})
    )
    cache = Path(cfg.data.cache_dir) / f"bench-r{d.rank}"
    rows = load_split(mirror, cache, args.split)
    questions = vqa_questions(rows)[: args.limit]
    caps = caption_questions(rows, args.captions)
    mine, my_caps = questions[d.rank :: d.world], caps[d.rank :: d.world]
    log.info(
        "%s: %d VQA questions, %d captions (%d/%d on this rank)",
        args.split,
        len(questions),
        len(caps),
        len(mine),
        len(my_caps),
    )

    tok = get_tokenizer(cfg)
    renderer = Renderer(tok, cfg.images.size, cfg.train.max_len)
    images = ImageMaker(cfg, load_quant_spec(mirror, cache))
    loaded = load_model(cfg, d.device)
    loaded.model.eval()
    lm = LabelPositionLM(loaded.model, loaded.base).eval()
    adapter_meta: dict[str, Any] | None = None
    args.out.mkdir(parents=True, exist_ok=True)

    for run in runs:
        kind, comps, label = RUNS[run]
        label = label.format(model=cfg.model.name.split("/")[-1])
        if kind == "fine-tuned" and adapter_meta is None:
            source = args.adapter or cfg.checkpoint.hub_repo
            if source is None:
                raise SystemExit("fine-tuned runs need --adapter or checkpoint.hub_repo")
            state, adapter_meta = fetch_adapter(
                source, Path(cfg.data.cache_dir) / f"adapter-r{d.rank}"
            )
            load_adapter_weights(loaded.model, {k: v.to(d.device) for k, v in state.items()})
        t0 = time.time()
        ev = ScoreEvaluator(renderer, images, comps, cfg.eval.batch_size, cfg.train.pad_to_multiple)
        recs = [r for part in d.gather_obj(ev.run(lm, mine, d.device)) for r in part]
        cap_recs: list[dict[str, Any]] = []
        if caps:
            mine_caps = generate_captions(
                loaded, tok, renderer, images, comps, my_caps, d.device, args.max_new_tokens
            )
            cap_recs = [r for part in d.gather_obj(mine_caps) for r in part]
        if not d.main:
            continue
        summary = summarize(recs)
        cal = calibration(recs)
        result: dict[str, Any] = {
            "run": run,
            "label": label,
            "kind": kind,
            "composites": list(comps),
            "split": args.split,
            "model": cfg.model.name,
            "summary": summary,
            "calibration": cal,
            "adapter": adapter_meta if kind == "fine-tuned" else None,
            "git": _git(),
            "seconds": round(time.time() - t0, 1),
            "image_size": cfg.images.size,
        }
        if cap_recs:
            from satquery.eval.metrics.caption import caption_scores

            preds = {str(r["qid"]): r["prediction"] for r in cap_recs}
            refs = {str(r["qid"]): [r["reference"]] for r in cap_recs}
            try:
                result["captions"] = caption_scores(preds, refs)
            except ImportError as exc:  # pycocoevalcap missing
                result["captions"] = {"n": len(cap_recs), "error": str(exc)}
        (args.out / f"bench_{run}.json").write_text(json.dumps(result, indent=1) + "\n")
        with gzip.open(args.out / f"bench_{run}.records.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in recs + cap_recs:
                fh.write(json.dumps(r) + "\n")
        (args.out / f"reliability_{run}.svg").write_text(
            reliability_svg(cal, f"({run}) {label}"), encoding="utf-8"
        )
        log.info(
            "(%s) %s: binary OA %.2f, MCQ OA %.2f",
            run,
            label,
            summary.get("binary", {}).get("OA", 0),
            summary.get("mcq", {}).get("OA", 0),
        )
        write_results_md(args.out)
    d.barrier()
    d.close()


if __name__ == "__main__":
    main()
