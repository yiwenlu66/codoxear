from __future__ import annotations

from pathlib import Path
from typing import Any

from .cc_log import cc_token_observation
from .pi_log import pi_context_token_update
from .pi_log import pi_token_update
from .rollout_jsonl import _read_jsonl_tail
from .token_signal import TOKEN_NONE
from .token_signal import TokenObservation
from .token_signal import token_update_observation


def _extract_token_observation(objs: list[dict[str, Any]]) -> TokenObservation:
    # Prefer the newest token signal in this batch.  A CC assistant usage row
    # with an unknown model is an explicit clear signal; do not scan behind it
    # to resurrect older known-model pressure.
    for obj in reversed(objs):
        pi_token = pi_token_update(obj)
        if pi_token is not None:
            return token_update_observation(pi_token)
        cc_observation = cc_token_observation(obj)
        if cc_observation.observed:
            return cc_observation
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
        return token_update_observation(
            pi_context_token_update(
                context_window=ctx,
                tokens_in_context=tt,
                as_of=obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
            )
        )
    return TOKEN_NONE


def _extract_token_update(objs: list[dict[str, Any]]) -> dict[str, Any] | None:
    return _extract_token_observation(objs).public_token


def _find_latest_token_observation(log_path: Path, max_scan_bytes: int = 32 * 1024 * 1024) -> TokenObservation:
    scan = min(256 * 1024, max_scan_bytes)
    if scan <= 0:
        return TOKEN_NONE
    while scan <= max_scan_bytes:
        observation = _extract_token_observation(_read_jsonl_tail(log_path, scan))
        if observation.observed:
            return observation
        scan *= 2
    return TOKEN_NONE


def _find_latest_token_update(log_path: Path, max_scan_bytes: int = 32 * 1024 * 1024) -> dict[str, Any] | None:
    return _find_latest_token_observation(log_path, max_scan_bytes=max_scan_bytes).public_token


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
