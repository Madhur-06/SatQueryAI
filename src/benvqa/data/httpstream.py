"""Read huge remote archives as one forward-only stream that survives connection drops.

``ResumableHTTPStream`` reconnects with ``Range: bytes=<consumed>-`` when the connection
breaks, and keeps feeding the *same* downstream decompressor/tar reader, which never
notice the interruption.

``ParallelRangeReader`` is the default for big downloads. Zenodo throttles each connection
to about 0.4 MB/s (measured 2026-10-11: 4 parallel connections gave 4 x 0.41 MB/s). It
therefore fetches consecutive chunks over N connections, with bounded look-ahead, and
serves them in order. Every chunk is its own Range request with retries, so drops never
reach the consumer.

Local paths are opened directly (tests, local copies).
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO

import httpx

log = logging.getLogger(__name__)
RETRYABLE = (httpx.TransportError, httpx.RemoteProtocolError, ConnectionError, TimeoutError)


class ResumableHTTPStream(io.RawIOBase):
    def __init__(
        self,
        url: str,
        *,
        start: int = 0,
        chunk_size: int = 1 << 20,
        max_retries: int = 50,
        timeout: float = 60.0,
        backoff: float = 2.0,
        max_backoff: float = 120.0,
    ) -> None:
        super().__init__()
        self.url = url
        self.pos = start
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff, self.max_backoff = backoff, max_backoff
        self.reconnects = 0
        self.total_size: int | None = None
        self._client = httpx.Client(follow_redirects=True, timeout=timeout)
        self._resp: httpx.Response | None = None
        self._it: Iterator[bytes] | None = None
        self._buf = b""
        self._eof = False

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        req = self._client.build_request("GET", self.url, headers=headers)
        resp = self._client.send(req, stream=True)
        if resp.status_code == 416:  # nothing left
            resp.close()
            self._eof = True
            return
        resp.raise_for_status()
        if self.pos and resp.status_code != 206:
            resp.close()
            raise RuntimeError(f"server ignored Range (status {resp.status_code}); cannot resume")
        if self.pos:
            content_range = resp.headers.get("content-range", "")
            if not content_range.startswith(f"bytes {self.pos}-"):
                resp.close()
                raise RuntimeError(f"unexpected Content-Range {content_range!r} at {self.pos}")
            self.total_size = int(content_range.rsplit("/", 1)[-1])
        elif "content-length" in resp.headers:
            self.total_size = int(resp.headers["content-length"])
        self._resp = resp
        self._it = resp.iter_raw(self.chunk_size)

    def _close_resp(self) -> None:
        if self._resp is not None:
            self._resp.close()
        self._resp, self._it = None, None

    def _next_chunk(self) -> bytes:
        failures = 0
        while True:
            try:
                if self._it is None:
                    self._open()
                    if self._eof:
                        return b""
                assert self._it is not None
                chunk = next(self._it)
                if chunk:
                    return chunk
            except StopIteration:
                self._close_resp()
                if self.total_size is None or self.pos >= self.total_size:
                    self._eof = True
                    return b""
                failures += 1  # premature end of body: treat like a drop
                err: Exception = ConnectionError(f"stream ended early at {self.pos}")
            except RETRYABLE as exc:
                self._close_resp()
                failures += 1
                err = exc
            if failures > self.max_retries:
                raise ConnectionError(f"giving up on {self.url} at byte {self.pos}") from err
            self.reconnects += 1
            delay = min(self.max_backoff, self.backoff * 2 ** min(failures - 1, 10))
            log.warning(
                "stream dropped at byte %d (%s); reconnect %d in %.0fs",
                self.pos,
                err,
                self.reconnects,
                delay,
            )
            time.sleep(delay)

    def readinto(self, b) -> int:  # type: ignore[override]
        if not self._buf and not self._eof:
            self._buf = self._next_chunk()
        if not self._buf:
            return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self.pos += n
        return n

    def close(self) -> None:
        self._close_resp()
        self._client.close()
        super().close()


class ParallelRangeReader(io.RawIOBase):
    """Ordered stream over ``connections`` concurrent Range requests of ``chunk_size``."""

    def __init__(
        self,
        url: str,
        *,
        connections: int = 16,
        chunk_size: int = 32 << 20,
        lookahead: int | None = None,
        max_retries: int = 30,
        timeout: float = 60.0,
        backoff: float = 2.0,
        max_backoff: float = 120.0,
    ) -> None:
        super().__init__()
        self.url, self.chunk_size = url, chunk_size
        self.max_retries, self.backoff, self.max_backoff = max_retries, backoff, max_backoff
        self.retries = 0
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            limits=httpx.Limits(max_connections=connections + 2),
        )
        self.total_size = self._probe_size()
        self.n_chunks = -(-self.total_size // chunk_size)
        self._pool = ThreadPoolExecutor(connections, thread_name_prefix="range")
        self._lookahead = lookahead or connections + 2
        self._futures: dict[int, object] = {}
        self._next_submit = 0
        self._next_read = 0
        self._buf = memoryview(b"")
        self._lock = threading.Lock()
        self._closing = False
        self._fill()

    def readable(self) -> bool:
        return True

    def _probe_size(self) -> int:
        resp = self._client.get(self.url, headers={"Range": "bytes=0-0"})
        if resp.status_code != 206:
            raise RuntimeError(f"server does not support Range (status {resp.status_code})")
        return int(resp.headers["content-range"].rsplit("/", 1)[-1])

    def _fetch(self, idx: int) -> bytes:
        start = idx * self.chunk_size
        end = min(self.total_size, start + self.chunk_size) - 1
        parts: list[bytes] = []
        got, failures = 0, 0
        while start + got <= end:
            try:
                headers = {"Range": f"bytes={start + got}-{end}"}
                with self._client.stream("GET", self.url, headers=headers) as resp:
                    if resp.status_code != 206:
                        raise httpx.RemoteProtocolError(f"status {resp.status_code}")
                    for piece in resp.iter_raw(1 << 20):
                        parts.append(piece)
                        got += len(piece)
                if start + got <= end:
                    raise ConnectionError(f"short read at {start + got}")
            except (*RETRYABLE, httpx.HTTPStatusError) as exc:
                if self._closing:
                    return b""
                failures += 1
                with self._lock:
                    self.retries += 1
                if failures > self.max_retries:
                    raise ConnectionError(f"chunk {idx} failed {failures} times") from exc
                delay = min(self.max_backoff, self.backoff * 2 ** min(failures - 1, 10))
                log.warning("chunk %d dropped at +%d (%s); retry in %.0fs", idx, got, exc, delay)
                time.sleep(delay)
        data = b"".join(parts)
        return data[: end - start + 1]  # a server that ignores the range end sends extra

    def _fill(self) -> None:
        while self._next_submit < self.n_chunks and (
            self._next_submit - self._next_read < self._lookahead
        ):
            self._futures[self._next_submit] = self._pool.submit(self._fetch, self._next_submit)
            self._next_submit += 1

    def readinto(self, b) -> int:  # type: ignore[override]
        if not len(self._buf):
            if self._next_read >= self.n_chunks:
                return 0
            fut = self._futures.pop(self._next_read)
            self._buf = memoryview(fut.result())  # type: ignore[attr-defined]
            self._next_read += 1
            self._fill()
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n

    def close(self) -> None:
        self._closing = True
        for fut in self._futures.values():
            fut.cancel()  # type: ignore[attr-defined]
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()
        super().close()


def open_source(src: str | Path, connections: int = 16, **kwargs) -> IO[bytes]:
    """``http(s)://`` URL -> parallel range reader (``connections`` > 1) or a single
    resumable stream; anything else -> local file."""
    s = str(src)
    if s.startswith(("http://", "https://")):
        raw: io.RawIOBase = (
            ParallelRangeReader(s, connections=connections, **kwargs)
            if connections > 1
            else ResumableHTTPStream(s, **kwargs)
        )
        return io.BufferedReader(raw, buffer_size=1 << 20)
    return Path(s).open("rb")
