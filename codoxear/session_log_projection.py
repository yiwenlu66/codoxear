from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .token_signal import TOKEN_NONE
from .token_signal import TokenObservation


LogRevision = tuple[int, int, int, int]
LogIdentity = tuple[str, int, int]


def log_revision(path: Path) -> LogRevision | None:
    """Return the file identity plus append/truncation revision used by commits."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def log_identity(path: Path, revision: LogRevision) -> LogIdentity:
    return (str(path), int(revision[0]), int(revision[1]))


@dataclass(frozen=True)
class LogDerivedSessionObservation:
    """One ordered observation of registry fields derived from a JSONL log.

    ``log_path`` and ``revision`` identify the binding/file generation;
    ``start_off``/``end_off`` locate the observed byte range. Missing setting
    fields mean "no evidence". ``token`` retains the separate explicit-clear
    state required to clear an older cached token.

    ``effective_settings`` is reserved for a full refresh which has already
    reconciled live bridge/sidecar authority against log evidence. Incremental
    observations leave it false so the coordinator applies the established Pi
    live-effort priority itself.
    """

    log_path: Path
    revision: LogRevision
    start_off: int
    end_off: int
    token: TokenObservation = TOKEN_NONE
    model_provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    last_conversation_ts: float | None = None
    invalidate_idle_cache: bool = False
    history_scanned: bool = False
    settings_revision: LogRevision | None = None
    effective_settings: bool = False
    replace_settings: bool = False

    def __post_init__(self) -> None:
        start = int(self.start_off)
        end = int(self.end_off)
        if start < 0 or end < start:
            raise ValueError("invalid log observation byte range")
        if end > int(self.revision[2]):
            raise ValueError("log observation ends beyond its revision")
