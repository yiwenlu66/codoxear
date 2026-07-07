from __future__ import annotations

from pathlib import Path
from typing import Any

from .cc_log import cc_token_update
from .pi_log import pi_context_token_update
from .pi_log import pi_token_update
from .rollout_jsonl import _read_jsonl_tail


def _extract_token_update(objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Prefer the newest token_count in this batch.
    for obj in reversed(objs):
        pi_token = pi_token_update(obj)
        if pi_token is not None:
            return pi_token
        cc_token = cc_token_update(obj)
        if cc_token is not None:
            return cc_token
        if obj.get("type") != "event_msg":
            continue
        p = obj.get("payload")
        if not isinstance(p, dict):
            raise ValueError("invalid token_count payload")
        if p.get("type") != "token_count":
            continue
        info = p.get("info")
        if not isinstance(info, dict) or not isinstance(info.get("total_token_usage"), dict):
            continue
        ctx = info.get("model_context_window")
        last = info.get("last_token_usage")
        if not isinstance(ctx, int) or not isinstance(last, dict):
            continue
        tt = last.get("total_tokens")
        if not isinstance(tt, int):
            continue
        return pi_context_token_update(
            context_window=ctx,
            tokens_in_context=tt,
            as_of=obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
        )
    return None


def _find_latest_token_update(log_path: Path, max_scan_bytes: int = 32 * 1024 * 1024) -> dict[str, Any] | None:
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    while scan <= max_scan_bytes:
        token = _extract_token_update(_read_jsonl_tail(log_path, scan))
        if token is not None:
            return token
        scan *= 2
    return None


def _find_latest_turn_context(log_path: Path, max_scan_bytes: int = 8 * 1024 * 1024) -> dict[str, Any] | None:
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return None
    while scan <= max_scan_bytes:
        objs = _read_jsonl_tail(log_path, scan)
        for obj in reversed(objs):
            if not isinstance(obj, dict):
                continue
            if obj.get("type") != "turn_context":
                continue
            payload = obj.get("payload")
            if isinstance(payload, dict):
                return payload
        scan *= 2
    return None
