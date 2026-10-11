"""Data-loader throughput on real shards (CPU only): is data loading ever the bottleneck?

    PYTHONPATH=src python scripts/bench_throughput.py --config configs/smoke_08b.yaml \
        --batches 300 --workers 2

Prints conversations/s, QAs/s and tokens/s for one rank's DataLoader, the per-conversation
CPU cost of each stage in a single process, and the margin over what training needs
(``--need-conv-per-s``, default 1.5 per rank: about 2x the measured T4 step rate).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benvqa.config import load_config
from benvqa.data.stream import TrainStream, make_train_loader


def stage_costs(stream: TrainStream, n: int = 200) -> dict[str, float]:
    """Single-process cost per conversation of each stage (ms)."""
    stream._setup(0)
    sid = stream.stream_id(0)
    pos = stream.plan.locate(sid, 0)
    group = stream.plan.groups(sid, pos.epoch)[pos.group]
    t0 = time.perf_counter()
    tables = [stream._read(stream.mirror.download(s.path, stream.cache / "bench")) for s in group]
    t_read = (time.perf_counter() - t0) / sum(len(t["qa"]) for t in tables)
    rows = stream.plan.row_order(group, 0)[:n]
    t = {"read_shard": t_read * 1e3}
    t0 = time.perf_counter()
    convs = [stream._conversation(tables[j], r, 0) for j, r in rows]
    t["conversation"] = (time.perf_counter() - t0) / len(rows) * 1e3
    t0 = time.perf_counter()
    rendered = [stream.renderer.render(c) for c, _ in convs]
    t["render_tokenize"] = (time.perf_counter() - t0) / len(rows) * 1e3
    t0 = time.perf_counter()
    items = [stream._item(c, r) for c, r in zip(convs, rendered, strict=True)]
    t["images_patchify"] = (time.perf_counter() - t0) / len(rows) * 1e3
    from benvqa.data.collate import collate

    t0 = time.perf_counter()
    for i in range(0, len(items), stream.cfg.train.micro_batch):
        collate(items[i : i + stream.cfg.train.micro_batch], 0, stream.renderer.image_pad, {})
    t["collate"] = (time.perf_counter() - t0) / len(rows) * 1e3
    t["total"] = sum(t.values())
    return {k: round(v, 3) for k, v in t.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/smoke_08b.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--need-conv-per-s", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, [*args.set, f"data.num_workers={args.workers}"])

    costs = stage_costs(TrainStream(cfg, 0, 1))
    print("per-conversation CPU cost (ms, one process):", costs)

    loader = make_train_loader(TrainStream(cfg, 0, 1), pin_memory=False)
    it = iter(loader)
    next(it)  # worker start-up, first downloads
    t0 = time.perf_counter()
    conv = qa = tok = padded = 0
    for _ in range(args.batches):
        b = next(it)
        m = b["meta"]
        conv += m["n_conversations"]
        qa += m["n_qa"]
        tok += m["n_tokens"]
        padded += m["n_padded"]
    dt = time.perf_counter() - t0
    res = {
        "workers": args.workers,
        "micro_batch": cfg.train.micro_batch,
        "conversations_per_s": round(conv / dt, 1),
        "qa_per_s": round(qa / dt, 1),
        "tokens_per_s": round(tok / dt),
        "qa_per_conversation": round(qa / conv, 2),
        "tokens_per_conversation": round(tok / conv, 1),
        "padding_fraction": round(1 - tok / padded, 4),
        "margin_over_need": round(conv / dt / args.need_conv_per_s, 1),
        "stage_ms": costs,
        "conversation_cost_ms_p50": float(np.round(costs["total"], 2)),
    }
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
