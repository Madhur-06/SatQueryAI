"""M5a numerical checks on a real Qwen3.5 checkpoint (default Qwen3.5-0.8B), real shards.

    PYTHONPATH=src python scripts/smoke_model_checks.py --config configs/smoke_08b.yaml

1. Label-position-only loss == naive full-logits loss (fp32), tolerance 1e-3.
2. fp16 vs fp32: max |delta| of the answer-position log-probs, and loss difference.
3. Right padding: hidden states at real positions of a padded batch == each sample alone.
4. Forward/backward time + peak memory for one micro-batch (fp16 autocast).
Writes ``reports/smoke_<model>_checks.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benvqa.config import load_config
from benvqa.data.stream import TrainStream
from benvqa.model.load import fast_kernel_report, load_model
from benvqa.model.loss import LabelPositionLM, reference_loss


def batches(cfg, n):  # type: ignore[no-untyped-def]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
    it = iter(TrainStream(cfg.model_copy(deep=True), 0, 1, tokenizer=tok))
    return [next(it) for _ in range(n)]


def to(b, dev):  # type: ignore[no-untyped-def]
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/smoke_08b.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {
        "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
        "kernels": fast_kernel_report(),
    }
    cfg32 = load_config(args.config, [*args.set, "model.dtype=float32", "data.num_workers=0"])
    bs = [to(b, dev) for b in batches(cfg32, args.n)]
    out["model"] = cfg32.model.name

    loaded = load_model(cfg32, dev)
    with torch.no_grad():  # LoRA B is zero-initialised; perturb so the adapter matters
        for p in loaded.model.parameters():
            if p.requires_grad:
                p.add_(1e-3 * torch.randn_like(p))
    lm = LabelPositionLM(loaded.model, loaded.base).eval()
    diffs, ours32 = [], []
    with torch.no_grad():
        for b in bs:
            ours, _ = lm(b, 1.0)
            ref = reference_loss(lm.base, b, 1.0)
            diffs.append(abs(float(ours) - float(ref)))
            ours32.append(float(ours))
    out["loss_equivalence"] = {
        "max_abs_diff": max(diffs),
        "ok": max(diffs) < 1e-3,
        "losses": ours32,
    }

    # right padding: batch vs singles
    b = next(x for x in bs if x["input_ids"].shape[0] > 1)
    pad_id = int(loaded.base.config.image_token_id)
    lens = b["attention_mask"].sum(1)
    per_img = int(b["image_grid_thw"][0].prod())
    n_img = [int((b["input_ids"][i] == pad_id).sum()) // (per_img // 4) for i in range(len(lens))]
    tiles = torch.split(b["pixel_tiles"], [n * per_img for n in n_img])
    grids = torch.split(b["image_grid_thw"], n_img)
    worst = 0.0
    with torch.no_grad():
        hb = lm.hidden_states(b, torch.float32)
        for i in range(len(lens)):
            n = int(lens[i])
            single = {
                "input_ids": b["input_ids"][i : i + 1, :n],
                "attention_mask": b["attention_mask"][i : i + 1, :n],
                "mm_token_type_ids": b["mm_token_type_ids"][i : i + 1, :n],
                "pixel_tiles": tiles[i],
                "image_grid_thw": grids[i],
            }
            h1 = lm.hidden_states(single, torch.float32)[0]
            worst = max(worst, float((hb[i, :n] - h1).abs().max() / h1.abs().max()))
    out["right_padding"] = {"lengths": lens.tolist(), "max_rel_diff": worst, "ok": worst < 1e-3}

    # fp16 vs fp32 (same adapter values, weights cast to fp16)
    del lm
    torch.cuda.empty_cache()
    state = {k: v.clone() for k, v in loaded.model.state_dict().items() if "lora_" in k}
    del loaded
    torch.cuda.empty_cache()
    cfg16 = load_config(args.config, [*args.set, "data.num_workers=0"])
    l16 = load_model(cfg16, dev)
    l16.model.load_state_dict(state, strict=False)
    lm16 = LabelPositionLM(l16.model, l16.base).eval()
    d16 = []
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        for b, l32 in zip(bs, ours32, strict=True):
            loss16, _ = lm16(b, 1.0)
            d16.append(abs(float(loss16) - l32))
    out["fp16_vs_fp32"] = {
        "max_abs_loss_diff": max(d16),
        "finite": all(map(lambda x: x == x, d16)),
        "ok": max(d16) < 0.05,
    }

    # speed / memory of one training micro-batch in fp16
    lm16.train()
    params = [p for p in lm16.parameters() if p.requires_grad]
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    times = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for b in bs:
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            loss, _ = lm16(b, float(b["turn_weight"].sum()))
        scaler.scale(loss).backward()
        for p in params:
            p.grad = None
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    out["train_micro_batch"] = {
        "seconds": [round(t, 3) for t in times],
        "conversations": bs[0]["input_ids"].shape[0],
        "tokens": [int(b["attention_mask"].sum()) for b in bs],
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if dev.type == "cuda"
        else None,
        "trainable_params": l16.info["trainable"],
    }
    print(json.dumps(out, indent=1))
    name = cfg32.model.name.split("/")[-1].lower().replace(".", "")
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/smoke_{name}_checks.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
