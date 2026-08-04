from __future__ import annotations

import json
from pathlib import Path

from codoxear.message_routes import _read_chat_export_events
from codoxear.rollout_log import _read_chat_history_page
from codoxear.rollout_log import _read_chat_tail_page
from codoxear.transcript_search import search_chat_log_bounded


def _pi_user_post(text: str, **transport: object) -> dict[str, object]:
    return {
        "type": "message",
        **transport,
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def _write_posts(path: Path, posts: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(post, separators=(",", ":")) + "\n" for post in posts),
        encoding="utf-8",
    )


def _texts(events: list[dict[str, object]]) -> list[str]:
    return [event["text"] for event in events if isinstance(event.get("text"), str)]


def test_intercom_harness_and_test_posts_are_excluded_from_every_transcript_projection(tmp_path: Path) -> None:
    log_path = tmp_path / "pi-session.jsonl"
    intercom_post = _pi_user_post("intercom traffic", source="intercom")
    intercom_delivery_post = _pi_user_post("**📨 From supervisor**\n\nintercom delivery")
    harness_post = _pi_user_post("harness traffic", metadata={"tags": ["harness"]})
    test_post = _pi_user_post("test traffic", tag="test")
    real_user_post = _pi_user_post("real user message")
    _write_posts(log_path, [intercom_post, intercom_delivery_post, harness_post, test_post, real_user_post])

    tail, _before, _after, _has_older = _read_chat_tail_page(log_path, limit=20)
    history, _next_before, _history_has_older = _read_chat_history_page(
        log_path,
        before_byte=log_path.stat().st_size,
        limit=20,
    )
    exported = _read_chat_export_events(log_path, max_bytes=1024 * 1024)
    intercom_count, intercom_matches, _intercom_truncated = search_chat_log_bounded(log_path, "intercom")
    delivery_count, delivery_matches, _delivery_truncated = search_chat_log_bounded(log_path, "delivery")
    harness_count, harness_matches, _harness_truncated = search_chat_log_bounded(log_path, "harness")
    test_count, test_matches, _test_truncated = search_chat_log_bounded(log_path, "test")
    real_count, real_matches, _real_truncated = search_chat_log_bounded(log_path, "real user")

    # The POSTs reach the shared normalizer, but only human conversation is
    # recordable in transcript tail/history/export/search projections.
    assert _texts(tail) == ["real user message"]
    assert _texts(history) == ["real user message"]
    assert _texts(exported) == ["real user message"]
    assert (intercom_count, intercom_matches) == (0, [])
    assert (delivery_count, delivery_matches) == (0, [])
    assert (harness_count, harness_matches) == (0, [])
    assert (test_count, test_matches) == (0, [])
    assert real_count == 1
    assert _texts(real_matches) == ["real user message"]
