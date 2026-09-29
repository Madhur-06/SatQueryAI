"""Explicit, opt-in benchmark downloads (``satquery-eval --download <name>``).

Only sources confirmed with the team are here (see ``sources.py``):

* ``rsvqa-lr``: Zenodo record 6344334. File list and MD5 checksums are read from the
  Zenodo API at download time, and every file is verified against them.
* ``cdvqa``: annotation JSONs from github.com/YZHJessica/CDVQA, plus the SECOND train set
  (the 2,968 public pairs CDVQA was built from), as published on the SECOND authors'
  page https://captain-whu.github.io/SCD/ (Google Drive, 2.41 GB RAR, extracted
  with the system ``tar``, i.e. libarchive).

Downloads stream to ``<file>.part`` and are renamed only when complete and verified;
existing verified files are skipped, so re-running resumes cleanly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "satquery-eval/0.1 (+https://github.com)"

ZENODO_RSVQA_LR = "https://zenodo.org/api/records/6344334"
RSVQA_LR_SPLIT_FILES = [
    f"LR_split_{split}_{kind}.json"
    for split in ("train", "val", "test")
    for kind in ("questions", "answers", "images")
]

CDVQA_RAW = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
CDVQA_ANNOTATIONS = [
    f"{split}_{kind}.json"
    for split in ("Train", "Val", "Test", "Test2")
    for kind in ("questions", "answers", "images")
]
SECOND_TRAIN_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc&export=download&confirm=t"
)
SECOND_TRAIN_NAME = "SECOND_train_set.rar"
SECOND_TRAIN_SIZE = 2_406_111_691


def _open(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch(
    url: str,
    dest: Path,
    *,
    size: int | None = None,
    md5: str | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Download ``url`` to ``dest`` unless a verified copy already exists."""

    def verified(path: Path) -> bool:
        if size is not None and path.stat().st_size != size:
            return False
        return md5 is None or _md5(path) == md5

    if dest.exists() and verified(dest):
        log(f"  ok      {dest.name} (already present)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log(f"  fetch   {dest.name}" + (f" ({size / 1e6:,.1f} MB)" if size else ""))
    with _open(url) as response, part.open("wb") as fh:
        total, next_report = 0, 250 * CHUNK
        while block := response.read(CHUNK):
            fh.write(block)
            total += len(block)
            if total >= next_report:
                pct = f" {100 * total / size:.0f}%" if size else ""
                log(f"          {total / 1e6:,.0f} MB{pct}")
                next_report += 250 * CHUNK
    if not verified(part):
        got = part.stat().st_size
        part.unlink()
        raise OSError(f"{dest.name}: verification failed (size {got}, expected {size}, md5 {md5})")
    part.replace(dest)
    return dest


def download_rsvqa_lr(root: Path, log: Callable[[str], None] = print) -> Path:
    log(f"RSVQA-LR -> {root}")
    with _open(ZENODO_RSVQA_LR) as response:
        record = json.load(response)
    files = {f["key"]: f for f in record["files"]}
    for name in [*RSVQA_LR_SPLIT_FILES, "Images_LR.zip"]:
        info = files[name]
        algo, _, checksum = info["checksum"].partition(":")
        fetch(
            info["links"]["self"],
            root / name,
            size=int(info["size"]),
            md5=checksum if algo == "md5" else None,
            log=log,
        )
    images = root / "Images_LR"
    if not images.is_dir() or not any(images.glob("*.tif")):
        log("  unzip   Images_LR.zip")
        with zipfile.ZipFile(root / "Images_LR.zip") as zf:
            zf.extractall(root)
    log(f"  done: {sum(1 for _ in images.glob('*.tif'))} images")
    return root


def libarchive_tar() -> str | None:
    """A tar that can read RAR: bsdtar/libarchive (GNU tar, e.g. Git's, cannot)."""
    candidates = [str(Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "tar.exe")]
    candidates += [c for c in (shutil.which("bsdtar"), shutil.which("tar")) if c]
    for candidate in candidates:
        try:
            version = subprocess.run(
                [candidate, "--version"], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "libarchive" in version or "bsdtar" in version:
            return candidate
    return None


def _find_pair_dirs(root: Path) -> tuple[Path, Path] | None:
    for im1 in sorted(root.rglob("im1")):
        im2 = im1.parent / "im2"
        if im1.is_dir() and im2.is_dir():
            return im1, im2
    return None


def download_cdvqa(root: Path, log: Callable[[str], None] = print) -> Path:
    log(f"CDVQA -> {root}")
    for name in CDVQA_ANNOTATIONS:
        fetch(f"{CDVQA_RAW}/{name}", root / name, log=log)
    if _find_pair_dirs(root) is None:
        archive = fetch(
            SECOND_TRAIN_URL, root / "raw" / SECOND_TRAIN_NAME, size=SECOND_TRAIN_SIZE, log=log
        )
        tar = libarchive_tar()
        if tar is None:
            raise OSError(
                "Extracting the SECOND RAR needs a libarchive 'tar' (bsdtar; built into "
                "Windows 10+ as System32\\tar.exe) or another RAR extractor."
            )
        log(f"  extract SECOND_train_set.rar ({tar})")
        try:
            subprocess.run([tar, "-xf", str(archive), "-C", str(root / "raw")], check=True)
        except subprocess.CalledProcessError as exc:
            raise OSError(f"extracting {archive.name} failed: {exc}") from exc
    pair = _find_pair_dirs(root)
    if pair is None:
        raise OSError(f"No im1/im2 folders found under {root} after extraction.")
    for src, name in zip(pair, ("im1", "im2"), strict=True):
        link = root / name
        if not link.exists():
            src.rename(link)  # move to the layout the loader expects
    log(f"  done: {sum(1 for _ in (root / 'im1').glob('*.png'))} image pairs")
    return root


DOWNLOADERS: dict[str, Callable[..., Path]] = {
    "rsvqa-lr": download_rsvqa_lr,
    "cdvqa": download_cdvqa,
}
