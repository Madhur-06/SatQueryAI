"""``satquery-eval``: run a tool (or the full pipeline) on a benchmark subset.

Examples::

    satquery-eval --list
    satquery-eval --benchmark rsvqa-lr --tool vlm_vqa --subset 500
    satquery-eval --benchmark vrsbench-grounding --tool grounding --data-root D:/vrsbench
    satquery-eval --benchmark rsvqa-lr --tool pipeline --subset 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from satquery.eval.datasets import DATASETS, DEFAULT_ROOTS, DatasetNotFoundError, load_dataset
from satquery.eval.report import write_report
from satquery.eval.runner import PIPELINE, run_eval
from satquery.eval.sources import SOURCES
from satquery.tools.registry import default_registry


def _list() -> None:
    print("Benchmarks (default data root):")
    for name in sorted(DATASETS):
        print(f"  {name:<20} {DEFAULT_ROOTS[name]}")
    print("\nSources (nothing is downloaded automatically):")
    for key, src in SOURCES.items():
        print(f"  {key}: {src.homepage}  [{src.license}]")
    print("\nTools:")
    for spec in default_registry().export_specs():
        flags = " (mock)" if spec.is_mock else ""
        flags += "" if spec.available else f" (unavailable: {spec.unavailable_reason})"
        print(f"  {spec.name:<22} tasks={','.join(spec.tasks)}{flags}")
    print(f"  {PIPELINE:<22} the full agentic pipeline (classifier + planner + tools)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="satquery-eval", description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="list benchmarks, sources and tools")
    parser.add_argument("--benchmark", choices=sorted(DATASETS))
    parser.add_argument("--tool", help=f"registry tool name, or '{PIPELINE}'")
    parser.add_argument("--subset", type=int, default=None, help="random subset size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default=None, help="dataset split (default per benchmark)")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None, help="bigearthnet-txt only")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    if args.list:
        _list()
        return 0
    if not args.benchmark or not args.tool:
        parser.error("--benchmark and --tool are required (or use --list)")

    kwargs = {}
    if args.split:
        kwargs["split"] = args.split
    if args.image_root and args.benchmark == "bigearthnet-txt":
        kwargs["image_root"] = args.image_root
    try:
        dataset = load_dataset(args.benchmark, args.data_root, **kwargs)
        report = run_eval(dataset, args.tool, subset=args.subset, seed=args.seed, progress=True)
    except DatasetNotFoundError as exc:
        print(f"Dataset not found.\n{exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = write_report(report, args.out)
    headline = {k: v for k, v in report.metrics.items() if not isinstance(v, dict)}
    print(f"{report.n_samples} samples, {report.n_errors} errors; {headline}")
    print(f"wrote {json_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
