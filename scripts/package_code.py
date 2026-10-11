"""Zip the code the Kaggle notebooks need (upload as a Kaggle Dataset named "benvqa-code").

    uv run --no-sync python scripts/package_code.py   # -> dist/benvqa-code.zip

Only committed files are packaged (``git ls-files``), plus ``CODE_VERSION`` with the commit,
so a notebook run is always traceable to a commit.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("src/benvqa/", "configs/", "scripts/", "requirements-kaggle.txt", "pyproject.toml")


def main() -> None:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()  # fmt: skip
    sha = subprocess.run(["git", "log", "--oneline", "-1"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()  # fmt: skip
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *INCLUDE], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()  # fmt: skip
    out = ROOT / "dist" / "benvqa-code.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.startswith(INCLUDE):
                zf.write(ROOT / f, f)
                n += 1
        zf.writestr(
            "CODE_VERSION", sha + (" (+uncommitted changes NOT included)" if dirty else "") + "\n"
        )
    print(f"wrote {out} ({n} files, {sha})")
    if dirty:
        print("warning: uncommitted changes are not included:\n" + dirty)


if __name__ == "__main__":
    main()
