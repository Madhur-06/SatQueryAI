"""Per-request storage: ``data_dir/requests/<request_id>/{inputs,outputs}``.

Request ids are uuid4 hex strings and file names are sanitised, so user input can never
address a path outside the request directory.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import BinaryIO

from satquery.errors import ErrorCode, SatQueryError

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
CHUNK = 1 << 20


def safe_filename(name: str, fallback: str = "upload") -> str:
    base = Path(name.replace("\\", "/")).name  # drop any client-supplied directories
    cleaned = _UNSAFE.sub("_", base).strip("._")
    return cleaned[:120] or fallback


class RequestStore:
    def __init__(self, data_dir: Path, max_upload_bytes: int) -> None:
        self.root = (data_dir / "requests").resolve()
        self.max_upload_bytes = max_upload_bytes

    def new_request(self) -> tuple[str, Path]:
        request_id = uuid.uuid4().hex
        path = self.root / request_id
        (path / "inputs").mkdir(parents=True)
        (path / "outputs").mkdir()
        return request_id, path

    def save_upload(self, request_dir: Path, index: int, filename: str, stream: BinaryIO) -> Path:
        dest = request_dir / "inputs" / f"{index}_{safe_filename(filename)}"
        written = 0
        with dest.open("wb") as fh:
            while chunk := stream.read(CHUNK):
                written += len(chunk)
                if written > self.max_upload_bytes:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise SatQueryError(
                        ErrorCode.UPLOAD_TOO_LARGE,
                        f"{filename}: larger than the upload limit "
                        f"({self.max_upload_bytes // (1 << 20)} MB).",
                        details={"file": filename, "limit_bytes": self.max_upload_bytes},
                    )
                fh.write(chunk)
        return dest

    def output_file(self, request_id: str, name: str) -> Path | None:
        """Resolve an output file, or None if the id/name is invalid or missing."""
        if not _ID_RE.match(request_id) or safe_filename(name) != name:
            return None
        path = (self.root / request_id / "outputs" / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return None
        return path
