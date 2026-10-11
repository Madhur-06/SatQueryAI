"""Kaggle session limits: stop cleanly before the 12 h kill, and on SIGTERM / SIGINT.

``SESSION_START_EPOCH`` is set in the notebook's very first cell (before pip installs), so
the budget covers set-up time too. Without it, the guard counts from process start.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
import time

log = logging.getLogger(__name__)


class SessionGuard:
    def __init__(self, max_hours: float, start_epoch: float | None = None) -> None:
        env = os.environ.get("SESSION_START_EPOCH")
        self.start = start_epoch if start_epoch is not None else float(env) if env else time.time()
        self.deadline = self.start + max_hours * 3600
        self.signal_received: str | None = None
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                with contextlib.suppress(ValueError, OSError):  # platform specific
                    signal.signal(sig, self._handle)

    def _handle(self, signum: int, frame: object) -> None:
        name = signal.Signals(signum).name
        if self.signal_received is not None and name == "SIGINT":
            raise KeyboardInterrupt  # second Ctrl-C: really stop
        log.warning("%s received: checkpoint and stop at the end of this step", name)
        self.signal_received = name

    def remaining_s(self) -> float:
        return self.deadline - time.time()

    def elapsed_s(self) -> float:
        return time.time() - self.start

    def stop_reason(self) -> str | None:
        if self.signal_received:
            return f"signal {self.signal_received}"
        if self.remaining_s() <= 0:
            return "session time limit"
        return None
