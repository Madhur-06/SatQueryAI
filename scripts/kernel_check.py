"""Fast GatedDeltaNet kernels (fla / Triton) vs the torch reference path: speed + agreement.

    PYTHONPATH=src python scripts/kernel_check.py --config configs/smoke_08b.yaml

Same model, adapter and micro-batch; times fwd+bwd and compares loss and LoRA gradients.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M  # noqa: N812

from benvqa.config import load_config
from benvqa.data.stream import TrainStream
from benvqa.model.load import fast_kernel_report, load_model
from benvqa.model.loss import LabelPositionLM

KERNELS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)


def run(lm, b, params, n=3):  # type: ignore[no-untyped-def]
    times, loss = [], None
    for _ in range(n):
        for p in params:
            p.grad = None
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = lm(b, float(b["turn_weight"].sum()))
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    grads = torch.cat([p.grad.flatten().float() for p in params if p.grad is not None])
    return loss.item(), grads, min(times[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/smoke_08b.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    from transformers import AutoTokenizer

    cfg = load_config(args.config, [*args.set, "data.num_workers=0"])
    tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
    b = next(iter(TrainStream(cfg, 0, 1, tokenizer=tok)))
    b = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
    torch.manual_seed(0)
    loaded = load_model(cfg, torch.device("cuda"))
    with torch.no_grad():
        for p in loaded.model.parameters():
            if p.requires_grad:
                p.add_(1e-3 * torch.randn_like(p))
    lm = LabelPositionLM(loaded.model, loaded.base).train()
    params = [p for p in lm.parameters() if p.requires_grad]
    fast = {k: getattr(M, k) for k in KERNELS if hasattr(M, k)}
    l_fast, g_fast, t_fast = run(lm, b, params)
    for k, v in fast.items():  # torch reference path
        setattr(M, k, getattr(v, "__wrapped__", v))
    l_ref, g_ref, t_ref = run(lm, b, params)
    out = {
        "installed": fast_kernel_report(),
        "model": cfg.model.name,
        "tokens": int(b["attention_mask"].sum()),
        "conversations": int(b["input_ids"].shape[0]),
        "fwd_bwd_s": {
            "fast": round(t_fast, 3),
            "torch": round(t_ref, 3),
            "speedup": round(t_ref / t_fast, 2),
        },
        "loss": {"fast": l_fast, "torch": l_ref, "abs_diff": abs(l_fast - l_ref)},
        "grad_cosine": float(torch.nn.functional.cosine_similarity(g_fast, g_ref, dim=0)),
        "grad_rel_norm_diff": float((g_fast - g_ref).norm() / g_ref.norm()),
    }
    print(json.dumps(out, indent=1))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/kernel_check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
