"""Generate the Kaggle notebooks in ``notebooks/`` (edit here, then re-run; do not hand-edit).

uv run --no-sync python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def notebook(cells: list[dict], gpu: bool) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4" if gpu else "none",
                "isInternetEnabled": True,
                "isGpuEnabled": gpu,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SESSION_START = code("""
# Must be the very first cell: the session guard measures the 12 h limit from here.
import os, time
os.environ.setdefault("SESSION_START_EPOCH", str(time.time()))
print("session start", time.ctime(float(os.environ["SESSION_START_EPOCH"])))
""")

GET_CODE = code("""
# Code: either clone the repo (set REPO_URL; private repos need a GITHUB_TOKEN secret), or
# attach the zip made by `just kaggle-code` as a Kaggle Dataset named "benvqa-code".
import os, subprocess, zipfile, pathlib
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
def secret(name):
    try:
        return secrets.get_secret(name)
    except Exception:
        return None

os.environ["HF_TOKEN"] = secret("HF_TOKEN") or ""
assert os.environ["HF_TOKEN"], "add an HF_TOKEN secret (write access) in Add-ons -> Secrets"
if secret("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = secret("WANDB_API_KEY")

REPO_URL = ""  # e.g. "https://github.com/<you>/SAT_QueryAI.git"
CODE = pathlib.Path("/tmp/code")
if not CODE.exists():
    if REPO_URL:
        url = REPO_URL
        if secret("GITHUB_TOKEN"):
            url = url.replace("https://", f"https://{secret('GITHUB_TOKEN')}@")
        subprocess.run(["git", "clone", "--depth", "1", url, str(CODE)], check=True)
    else:
        zips = list(pathlib.Path("/kaggle/input").rglob("benvqa-code*.zip"))
        assert zips, "set REPO_URL or attach the benvqa-code dataset"
        zipfile.ZipFile(zips[0]).extractall(CODE)
os.chdir(CODE)
os.environ["PYTHONPATH"] = str(CODE / "src")
import sys; sys.path.insert(0, str(CODE / "src"))
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout
      or open(CODE / "CODE_VERSION").read())
""")

DISK = code("""
!df -h /kaggle/working /kaggle/tmp /tmp 2>/dev/null; nproc; free -g
""")


def build_notebook() -> dict:
    return notebook(
        [
            md("""
# benvqa Stage 0: build the BigEarthNet.txt shards (CPU session)

Streams BigEarthNet v2 S1/S2 from Zenodo (117.7 GB, never stored) and writes ~230 shards
of 1,000 images (~130 MB each, ~30 GB total) to the Hub dataset `<HF_USER>/benvqa-shards`.

* Accelerator: **None** (CPU). Internet: **on**. Secrets: `HF_TOKEN` (write).
* Resumable: if the session dies, just run the notebook again. Finished shards are
  skipped and the tars are re-streamed, decoding only the unfinished patches.
* Zenodo caps each connection at about 0.4 MB/s, so the builder opens 32 parallel range
  connections.
"""),
            SESSION_START,
            GET_CODE,
            code("""
!pip install -q "zstandard>=0.23" "tifffile>=2024.8" "huggingface_hub>=1.0" hf_xet \
    "httpx>=0.27" "pyarrow>=17" pyyaml
"""),
            DISK,
            code("""
from huggingface_hub import HfApi, hf_hub_download
HF_USER = HfApi().whoami()["name"]
REPO = f"{HF_USER}/benvqa-shards"
PRIVATE = False  # ~30 GB: a free account's private storage may not fit; the licence allows public
PARQUET = hf_hub_download("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", "BigEarthNet.txt.parquet",
                          repo_type="dataset", revision="72d865f2146f0a85b720f7f3ca1cdbaeafc3d316",
                          local_dir="/tmp/ann")
WORK = "/kaggle/tmp/build" if os.path.isdir("/kaggle/tmp") else "/tmp/build"
print(REPO, PARQUET, WORK)
"""),
            code("""
# ~2-4 h on Kaggle depending on Zenodo throughput. Progress: tail the log in another cell.
!python -m benvqa.data.build_shards --parquet {PARQUET} --mirror hub:{REPO} \\
    --work-dir {WORK} --connections 32 {"--private" if PRIVATE else ""} \\
    2>&1 | tee /kaggle/working/build_shards.log
"""),
            code("""
import json
report = json.load(open(f"{WORK}/build_report.json"))
print(json.dumps({k: report[k] for k in ("selected", "s1", "s2", "shards", "verify")}, indent=1))
assert str(report.get("verify", "")).startswith("ok"), "index does not match the selection"
"""),
        ],
        gpu=False,
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, nb in {"kaggle_build_shards.ipynb": build_notebook()}.items():
        (OUT / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
