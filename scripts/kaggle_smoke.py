"""M5b smoke test (Kaggle 2x T4, ~30 min): environment, micro-batch sweep, Hub round trip.

    python scripts/kaggle_smoke.py --config configs/kaggle_t4x2.yaml --nproc 2 \
        --set data.hub_dataset=hub:me/benvqa-shards --smoke-repo hub:me/benvqa-smoke \
        --hours-planned 45

1. Environment: torch / CUDA / GPUs, fast-kernel availability.
2. Sweep ``train.micro_batch`` (grad_accum keeps 32 conversations per step). Each setting
   runs ``--steps`` real DDP training steps. Reported: conversations/s and QAs/s (mean over
   the last half of the steps, after kernel compilation), peak GPU memory,
   ``data_wait_frac``, finite loss.
3. Hub checkpoint round trip with the best setting: train, checkpoint and mirror, delete
   the local checkpoints, rerun, and require "resumed from step N".
4. Recommend ``train.planned_images = conversations/s x 3600 x --hours-planned``.
Writes ``reports/m5b_smoke.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_report() -> dict:
    import torch

    from benvqa.model.load import fast_kernel_report

    out = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "kernels": fast_kernel_report(),
    }
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

        out["fla_import"] = "ok"
    except Exception as exc:  # report, don't fail: transformers falls back to torch
        out["fla_import"] = f"{type(exc).__name__}: {exc}"
    return out


def train(
    args: argparse.Namespace, run_dir: Path, sets: list[str], resume: str = "none"
) -> tuple[int, str]:
    cmd = ["-m", "benvqa.train", "--resume", resume, "--set", *sets]
    if args.config:
        cmd += ["--config", str(args.config)]
    if args.nproc > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(args.nproc),
            *cmd,
        ]
    else:
        cmd = [sys.executable, *cmd]
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "stdout.log"
    t0 = time.time()
    with log.open("w") as fh:
        code = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy()
        ).returncode
    print(f"  exit {code} after {time.time() - t0:.0f}s ({log})", flush=True)
    return code, log.read_text(errors="replace")


def summarize_log(path: Path, steps: int) -> dict:
    recs = [json.loads(line) for line in path.read_text().splitlines() if '"loss"' in line]
    if not recs:
        return {}
    tail = [r for r in recs if r["step"] > steps // 2] or recs[-1:]
    mean = lambda k: sum(r[k] for r in tail) / len(tail)  # noqa: E731
    return {
        "steps_logged": len(recs),
        "conversations_per_s": round(mean("conversations_per_s"), 3),
        "qa_per_s": round(mean("qa_per_s"), 2),
        "tokens_per_s": round(mean("tokens_per_s"), 1),
        "sec_per_step": round(mean("sec_per_step"), 2),
        "data_wait_frac": round(max(r["data_wait_frac"] for r in tail), 4),
        "peak_mem_gb": max(r["gpu_mem_gb"].get("peak", 0) for r in recs),
        "loss_first_last": [recs[0]["loss"], recs[-1]["loss"]],
        "finite": all(r["loss"] == r["loss"] for r in recs),
        "first_step_s": recs[0]["sec_per_step"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs/kaggle_t4x2.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--micro-batches", default="1,2,4")
    ap.add_argument("--conversations-per-step", type=int, default=32)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--smoke-repo", default=None, help="hub:<user>/<repo> for the round trip")
    ap.add_argument("--work", type=Path, default=Path("/tmp/benvqa_smoke"))
    ap.add_argument(
        "--hours-planned",
        type=float,
        default=45.0,
        help="2-GPU wall-clock hours of training you plan in total",
    )
    ap.add_argument("--out", type=Path, default=Path("reports/m5b_smoke.json"))
    args = ap.parse_args()
    report: dict = {"env": env_report(), "sweep": {}}
    print(json.dumps(report["env"], indent=1), flush=True)
    shutil.rmtree(args.work, ignore_errors=True)

    common = [
        *args.set,
        "train.log_every=1",
        "eval.every_steps=0",
        "eval.val_questions=32",
        f"train.max_steps={args.steps}",
        "checkpoint.every_steps=100000",
        "checkpoint.every_minutes=1000",
    ]
    for mb in (int(x) for x in args.micro_batches.split(",")):
        accum = max(1, args.conversations_per_step // (mb * args.nproc))
        run = args.work / f"mb{mb}"
        print(f"micro_batch={mb} grad_accum={accum}", flush=True)
        code, out = train(
            args,
            run,
            [
                *common,
                f"train.micro_batch={mb}",
                f"train.grad_accum={accum}",
                f"checkpoint.dir={run / 'ckpt'}",
                "checkpoint.hub_repo=null",
            ],
        )
        res = {"grad_accum": accum, "exit": code, "oom": "out of memory" in out.lower()}
        if (run / "train_log.jsonl").exists():
            res.update(summarize_log(run / "train_log.jsonl", args.steps))
        report["sweep"][mb] = res
        print(json.dumps(res), flush=True)

    ok = {mb: r for mb, r in report["sweep"].items() if r["exit"] == 0 and r.get("finite")}
    if not ok:
        report["verdict"] = "no micro-batch setting trained successfully - see the stdout logs"
    else:
        best = max(ok, key=lambda mb: ok[mb]["conversations_per_s"])
        rate = ok[best]["conversations_per_s"]
        report["best"] = {
            "micro_batch": best,
            "grad_accum": ok[best]["grad_accum"],
            "conversations_per_s": rate,
            "qa_per_s": ok[best]["qa_per_s"],
        }
        planned = int(rate * 3600 * args.hours_planned)
        report["recommend"] = {
            "planned_images": planned,
            "set": [
                f"train.micro_batch={best}",
                f"train.grad_accum={ok[best]['grad_accum']}",
                f"train.planned_images={planned}",
            ],
            "note": f"{rate:.2f} conv/s x {args.hours_planned} h; one 11.3 h session = "
            f"{int(rate * 3600 * 11.3):,} conversations",
        }
        if args.smoke_repo:
            run = args.work / "roundtrip"
            sets = [
                *common,
                f"train.micro_batch={best}",
                f"train.grad_accum={ok[best]['grad_accum']}",
                f"checkpoint.dir={run / 'ckpt'}",
                f"checkpoint.hub_repo={args.smoke_repo}",
                "checkpoint.every_steps=4",
                "train.max_steps=8",
            ]
            print("hub round trip: train 8 steps with checkpoints every 4", flush=True)
            c1, _ = train(args, run, sets, resume="auto")
            shutil.rmtree(run / "ckpt", ignore_errors=True)
            (run / "train_log.jsonl").unlink(missing_ok=True)
            print("local checkpoints deleted; resume from the Hub to step 10", flush=True)
            c2, out2 = train(args, run, [*sets[:-1], "train.max_steps=10"], resume="auto")
            report["roundtrip"] = {
                "first_exit": c1,
                "second_exit": c2,
                "resumed_from_hub": "resumed from step 8" in out2,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "env"}, indent=1, default=str))


if __name__ == "__main__":
    main()
