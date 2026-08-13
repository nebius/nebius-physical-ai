"""Shared, deterministic progress reporting for long NPA waits."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sys
import time
from typing import Callable


def _stderr(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")
    sys.stderr.flush()


_SECRET_DETAIL = re.compile(
    r"(?i)\b(access[_-]?key|api[_-]?key|secret|session[_-]?token|token|password)="
    r"[^\s;]+"
)
_ACCESS_ID = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


def _safe_detail(detail: str) -> str:
    cleaned = _SECRET_DETAIL.sub(r"\1=<redacted>", str(detail or ""))
    return _ACCESS_ID.sub("<redacted>", cleaned)


@dataclass
class WaitProgress:
    label: str
    interval: float = 15.0
    monotonic: Callable[[], float] = time.monotonic
    emit: Callable[[str], None] = _stderr
    _started: float = field(init=False, default=0.0)
    _last: float = field(init=False, default=0.0)

    def start(self, detail: str = "") -> None:
        self._started = self.monotonic()
        self._last = self._started
        safe = _safe_detail(detail)
        self.emit(f"{self.label}: started" + (f"; {safe}" if safe else ""))

    def tick(self, detail: str = "") -> None:
        now = self.monotonic()
        if now - self._last < self.interval:
            return
        self._last = now
        safe = _safe_detail(detail)
        self.emit(
            f"{self.label}: waiting {now - self._started:.0f}s"
            + (f"; {safe}" if safe else "")
        )

    def finish(self, outcome: str, detail: str = "") -> None:
        now = self.monotonic()
        safe = _safe_detail(detail)
        self.emit(
            f"{self.label}: {outcome} after {max(0.0, now - self._started):.0f}s"
            + (f"; {safe}" if safe else "")
        )
