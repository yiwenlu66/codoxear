import json
import unittest
import urllib.parse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from codoxear.message_cursor import decode_message_cursor
from codoxear.message_cursor import encode_message_cursor
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import _read_chat_export_events
from codoxear.message_routes import handle_messages_history
from codoxear.message_routes import handle_messages_search
from codoxear.message_routes import handle_messages_tail
from codoxear.rollout_log import _read_chat_history_page
from codoxear.session_model import Session
from codoxear.transcript_search import casefold_match_span
from codoxear.transcript_search import clip_search_match_text
from codoxear.transcript_search import clip_search_text_around_query
from codoxear.transcript_search import search_chat_events
from codoxear.transcript_search import search_chat_log
from codoxear.transcript_search import search_chat_log_bounded


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"


def _write_assistant_rows(path: Path, count: int) -> None:
    rows = []
    for i in range(count):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"a{i}"}],
                    "phase": "final_answer",
                },
                "ts": float(i),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestTranscriptExport(unittest.TestCase):
    def test_export_reads_all_chat_events_in_order(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 205)

            events = _read_chat_export_events(path, max_bytes=path.stat().st_size)

        self.assertEqual(len(events), 205)
        self.assertEqual(events[0].get("text"), "a0")
        self.assertEqual(events[-1].get("text"), "a204")
        self.assertIsInstance(events[0].get("_before_byte"), int)
        self.assertLess(events[0].get("_before_byte"), events[-1].get("_before_byte"))

    def test_export_rejects_oversized_logs_instead_of_truncating(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 2)
            with self.assertRaisesRegex(ValueError, "too large to export"):
                _read_chat_export_events(path, max_bytes=1)

    def test_streaming_search_can_bound_count_without_changing_default(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 25)

            exact_count, exact_matches = search_chat_log(path, "a", limit=1, max_line_bytes=4096)
            bounded_count, bounded_matches, truncated = search_chat_log_bounded(path, "a", limit=1, max_line_bytes=4096, count_limit=5)

        self.assertEqual(exact_count, 25)
        self.assertEqual(len(exact_matches), 1)
        self.assertEqual(bounded_count, 5)
        self.assertEqual(len(bounded_matches), 1)
        self.assertTrue(truncated)
        with self.assertRaisesRegex(ValueError, "count_limit is only supported"):
            search_chat_log_bounded(path, "a", limit=1, max_line_bytes=4096, count_limit=5, order="latest")

    def test_streaming_search_marks_count_truncated_when_oversized_record_skipped(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "x" * 200 + " needle " + "y" * 200}],
                    "phase": "final_answer",
                },
                "ts": 1.0,
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            count, matches, truncated = search_chat_log_bounded(path, "needle", limit=3, max_line_bytes=80)

        self.assertEqual(count, 0)
        self.assertEqual(matches, [])
        self.assertTrue(truncated)

    def test_streaming_search_oversized_skip_after_boundary_does_not_truncate_count(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            first = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle before boundary"}],
                    "phase": "final_answer",
                },
                "ts": 1.0,
            }
            oversized = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "x" * 200 + " needle after boundary"}],
                    "phase": "final_answer",
                },
                "ts": 2.0,
            }
            first_line = json.dumps(first) + "\n"
            path.write_text(first_line + json.dumps(oversized) + "\n", encoding="utf-8")
            before_oversized = len(first_line.encode("utf-8"))

            count, matches, truncated = search_chat_log_bounded(path, "needle", limit=3, max_line_bytes=256, before_byte=before_oversized)

        self.assertEqual(count, 1)
        self.assertEqual([match.get("text") for match in matches], ["needle before boundary"])
        self.assertFalse(truncated)

    def test_streaming_search_counts_logs_that_export_rejects(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 25)
            with self.assertRaisesRegex(ValueError, "too large to export"):
                _read_chat_export_events(path, max_bytes=1)

            count, matches = search_chat_log(path, "a2", limit=3, max_line_bytes=4096)

        self.assertEqual(count, 6)
        self.assertEqual([match.get("text") for match in matches], ["a2", "a20", "a21"])
        self.assertTrue(all(isinstance(match.get("_before_byte"), int) for match in matches))

    def test_streaming_search_can_return_latest_match_before_boundary(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            rows = []
            for idx in range(12):
                text = f"needle {idx}" if idx in {1, 5, 10} else f"plain {idx}"
                rows.append(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                            "phase": "final_answer",
                        },
                        "ts": float(idx),
                    }
                )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            _count, first_matches = search_chat_log(path, "needle", limit=3, max_line_bytes=4096)
            before_newest = first_matches[-1]["_before_byte"]

            count, matches = search_chat_log(path, "needle", limit=1, max_line_bytes=4096, before_byte=before_newest, order="latest")

        self.assertEqual(count, 2)
        self.assertEqual([match["text"] for match in matches], ["needle 5"])

    def test_streaming_search_ignores_unterminated_final_record(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle without newline"}],
                    "phase": "final_answer",
                },
                "ts": 1.0,
            }
            path.write_text(json.dumps(row), encoding="utf-8")

            count, matches = search_chat_log(path, "needle", limit=3, max_line_bytes=4096)

        self.assertEqual(count, 0)
        self.assertEqual(matches, [])

    def test_streaming_search_dedupes_assistant_repeats_across_chunks(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            row = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle repeat"}],
                    "phase": "final_answer",
                },
                "ts": 1.0,
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

            count, matches = search_chat_log(path, "needle", limit=10, max_line_bytes=4096)

        self.assertEqual(count, 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].get("text"), "needle repeat")

    def test_streaming_search_skips_malformed_records_without_stopping(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            good = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle later"}],
                    "phase": "final_answer",
                },
                "ts": 2.0,
            }
            path.write_text("{not-json}\n" + json.dumps(good) + "\n", encoding="utf-8")

            count, matches = search_chat_log(path, "needle", limit=5, max_line_bytes=4096)

        self.assertEqual(count, 1)
        self.assertEqual(matches[0].get("text"), "needle later")

    def test_streaming_search_skips_malformed_dict_records_without_stopping(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            bad = {"type": "response_item", "payload": "notdict"}
            good = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle after bad dict"}],
                    "phase": "final_answer",
                },
                "ts": 2.5,
            }
            path.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n", encoding="utf-8")

            count, matches = search_chat_log(path, "needle", limit=5, max_line_bytes=4096)

        self.assertEqual(count, 1)
        self.assertEqual(matches[0].get("text"), "needle after bad dict")

    def test_streaming_search_skips_huge_non_dict_json_without_stopping(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            good = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle after huge int"}],
                    "phase": "final_answer",
                },
                "ts": 2.7,
            }
            path.write_text(("1" * 5000) + "\n" + json.dumps(good) + "\n", encoding="utf-8")

            count, matches = search_chat_log(path, "needle", limit=5, max_line_bytes=10000)

        self.assertEqual(count, 1)
        self.assertEqual(matches[0].get("text"), "needle after huge int")

    def test_streaming_search_skips_deeply_nested_non_dict_json_without_stopping(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            good = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle after nested"}],
                    "phase": "final_answer",
                },
                "ts": 2.9,
            }
            nested = ("[" * 2000) + ("]" * 2000)
            path.write_text(nested + "\n" + json.dumps(good) + "\n", encoding="utf-8")

            count, matches = search_chat_log(path, "needle", limit=5, max_line_bytes=10000)

        self.assertEqual(count, 1)
        self.assertEqual(matches[0].get("text"), "needle after nested")

    def test_streaming_search_skips_oversized_records_without_buffering_rest(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            good = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "needle after oversized"}],
                    "phase": "final_answer",
                },
                "ts": 3.0,
            }
            path.write_bytes(b"x" * 2048 + b"\n" + (json.dumps(good) + "\n").encode("utf-8"))

            count, matches = search_chat_log(path, "needle", limit=5, max_line_bytes=1024)

        self.assertEqual(count, 1)
        self.assertEqual(matches[0].get("text"), "needle after oversized")

    def test_search_match_text_clipping_is_opt_in_and_query_centered(self) -> None:
        matches = [
            {"role": "assistant", "text": "x" * 40 + "needle" + "y" * 40, "_before_byte": 1},
            {"role": "user", "text": "xy", "_before_byte": 2},
        ]

        self.assertIs(clip_search_match_text(matches, 0, query="needle"), matches)
        clipped = clip_search_match_text(matches, 18, query="needle")

        self.assertLessEqual(len(clipped[0]["text"]), 18)
        self.assertIn("needle", clipped[0]["text"])
        self.assertFalse(clipped[0]["text"].startswith("…"))
        self.assertFalse(clipped[0]["text"].endswith("…"))
        self.assertTrue(clipped[0]["text_truncated"])
        self.assertEqual(clipped[0]["_before_byte"], 1)
        self.assertEqual(clipped[1]["text"], "xy")
        self.assertNotIn("text_truncated", clipped[1])
        self.assertEqual(matches[0]["text"], "x" * 40 + "needle" + "y" * 40)
        prefix, truncated = clip_search_text_around_query("abcdef", "missing", 3)
        self.assertEqual(prefix, "abc")
        self.assertTrue(truncated)

    def test_search_text_clipping_maps_casefold_offsets_to_original_text(self) -> None:
        self.assertEqual(casefold_match_span("ß" * 4 + "needle", "needle"), (4, 10))
        snippet, truncated = clip_search_text_around_query("ß" * 40 + "needle" + "y" * 40, "needle", 18)

        self.assertTrue(truncated)
        self.assertIn("needle", snippet)
        self.assertNotEqual(snippet, "y" * 18)
        exact, exact_truncated = clip_search_text_around_query("xxxneedlezzz", "needle", 6)
        self.assertEqual(exact, "needle")
        self.assertTrue(exact_truncated)

    def test_search_counts_all_matching_chat_events(self) -> None:
        events = [
            {"role": "user", "text": "Needle in first user turn", "_before_byte": 1},
            {"role": "assistant", "text": "no match", "_before_byte": 2},
            {"role": "assistant", "text": "another NEEDLE appears", "_before_byte": 3},
            {"role": "system", "text": "needle ignored", "_before_byte": 4},
        ]

        count, matches = search_chat_events(events, "needle", limit=1)

        self.assertEqual(count, 2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["_before_byte"], 1)

    def test_ui_has_copy_conversation_action(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "copyConversationBtn"', source)
        self.assertIn('function formatConversationForCopy(events)', source)
        self.assertIn('api(`/api/sessions/${sid}/messages/export`)', source)
        self.assertIn('title: "Copy conversation"', source)


# --- Route-handler tests (direct handler calls with injected dependencies) ---
#
# These replace the previous monkeypatch-based tests that patched server.MANAGER,
# server._require_auth, server._json_response, and server.TRANSCRIPT_SEARCH_MAX_LINE_BYTES.
# All four seams are now satisfied via MessageRouteDeps injection, mirroring the
# pattern established in tests/test_message_routes.py.

# Fixed HMAC secret so encode/decode are exercised against the real signing
# implementation, preserving the signed-cursor public contract under test.
_SECRET = b"test-transcript-route-secret"


class _FakeHandler:
    def __init__(self) -> None:
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True


def _session(td: str, log_path: Path | None) -> Session:
    return Session(
        session_id="s1",
        thread_id="thread-1",
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd=td,
        log_path=log_path,
        sock_path=Path(td) / "s1.sock",
    )


def _search_manager(session: object) -> SimpleNamespace:
    """Manager fake exposing only the methods handle_messages_search calls."""
    return SimpleNamespace(
        refresh_session_meta=lambda _sid: None,
        get_session=lambda _sid: session,
        _attach_notification_texts=lambda matches: matches,
    )


def _deps(**overrides):
    responses: list[tuple[int, dict[str, object]]] = []
    metrics: list[tuple[str, float]] = []

    def json_response(_handler, status: int, payload: dict[str, object]) -> None:
        responses.append((status, payload))

    def encode_cursor(*, kind: str, session, pos: int) -> str:
        return encode_message_cursor(kind=kind, session=session, pos=pos, secret=_SECRET)

    def decode_cursor(token: str, *, kind: str, session) -> int:
        return decode_message_cursor(token, kind=kind, session=session, secret=_SECRET)

    def runtime_snapshot(_sid: str, _session, **_kw):
        # state, busy, queue_len, token
        return {}, False, 0, None

    deps = MessageRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        launch_attempt_transcript_for_session_id=lambda _sid: None,
        transcript_export_max_bytes=50 * 1024 * 1024,
        transcript_search_max_line_bytes=64 * 1024,
        decode_message_cursor=decode_cursor,
        encode_message_cursor=encode_cursor,
        record_metric=lambda name, value: metrics.append((name, value)),
        message_runtime_snapshot=runtime_snapshot,
    )
    for name, value in overrides.items():
        object.__setattr__(deps, name, value)
    return deps, responses, metrics


def test_messages_search_route_applies_text_max_to_response_matches() -> None:
    with TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        text = "x" * 40 + "needle" + "y" * 40
        row = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                "phase": "final_answer",
            },
            "ts": 1.0,
        }
        second = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "second needle match"}],
                "phase": "final_answer",
            },
            "ts": 2.0,
        }
        log_path.write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
        session = _session(td, log_path)

        deps, responses, _metrics = _deps()
        handle_messages_search(
            _FakeHandler(),
            session_id="s1",
            query="q=needle&limit=1&text_max=18",
            manager=_search_manager(session),
            deps=deps,
        )

        assert len(responses) == 1
        status, body = responses[0]
        assert status == 200
        assert body["transcript_state"] == "bound"
        assert body["thread_id"] == "thread-1"
        assert body["log_path"] == str(log_path)
        assert body["match_count"] == 2
        assert len(body["matches"]) == 1
        match = body["matches"][0]
        assert len(match["text"]) <= 18
        assert "needle" in match["text"]
        assert match["text_truncated"] is True
        assert isinstance(match.get("_before_byte"), int)
        assert "_after_byte" not in match
        assert isinstance(match.get("history_cursor"), str)
        assert isinstance(match.get("load_cursor"), str)
        target_pos = decode_message_cursor(match["history_cursor"], kind="history", session=session, secret=_SECRET)
        assert target_pos == match["_before_byte"]
        load_pos = decode_message_cursor(match["load_cursor"], kind="history", session=session, secret=_SECRET)
        assert load_pos > match["_before_byte"]
        window_events, _next_before, _has_older = _read_chat_history_page(log_path, before_byte=load_pos, limit=20)
        assert window_events[-1]["text"] == text


def test_messages_search_route_marks_truncated_when_oversized_record_skipped() -> None:
    with TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        oversized = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x" * 200 + " needle hidden by line cap"}],
                "phase": "final_answer",
            },
            "ts": 1.0,
        }
        log_path.write_text(json.dumps(oversized) + "\n", encoding="utf-8")
        session = _session(td, log_path)

        # transcript_search_max_line_bytes is injected via deps (no module patch).
        deps, responses, _metrics = _deps(transcript_search_max_line_bytes=80)
        handle_messages_search(
            _FakeHandler(),
            session_id="s1",
            query="q=needle&limit=1",
            manager=_search_manager(session),
            deps=deps,
        )

    assert len(responses) == 1
    status, body = responses[0]
    assert status == 200
    assert body["match_count"] == 0
    assert body["matches"] == []
    assert body["match_count_truncated"] is True


def test_messages_search_route_can_bound_match_count() -> None:
    with TemporaryDirectory() as td:
        log_path = Path(td) / "rollout.jsonl"
        _write_assistant_rows(log_path, 12)
        session = _session(td, log_path)

        deps, responses, _metrics = _deps()
        handle_messages_search(
            _FakeHandler(),
            session_id="s1",
            query="q=a&limit=1&count_max=5",
            manager=_search_manager(session),
            deps=deps,
        )

    assert len(responses) == 1
    status, body = responses[0]
    assert status == 200
    assert body["match_count"] == 5
    assert body["match_count_truncated"] is True
    assert len(body["matches"]) == 1


def test_messages_search_rejects_count_max_with_latest_order() -> None:
    deps, responses, _metrics = _deps()
    handle_messages_search(
        _FakeHandler(),
        session_id="s1",
        query="q=needle&count_max=5&order=latest",
        manager=SimpleNamespace(
            refresh_session_meta=lambda _sid: None,
            get_session=lambda _sid: object(),
        ),
        deps=deps,
    )

    assert responses == [(400, {"error": "count_max is only supported with order=first"})]


def test_messages_search_rejects_malformed_text_max() -> None:
    deps, responses, _metrics = _deps()
    handle_messages_search(
        _FakeHandler(),
        session_id="s1",
        query="q=needle&text_max=bad",
        manager=SimpleNamespace(
            refresh_session_meta=lambda _sid: None,
            get_session=lambda _sid: object(),
        ),
        deps=deps,
    )

    assert responses == [(400, {"error": "text_max must be an integer"})]


def test_message_routes_reject_malformed_limits() -> None:
    # Preserves the four-subtest coverage of the original: search limit,
    # search count_max, tail limit, and history limit all reject non-integers.
    cases = [
        (
            "search limit",
            handle_messages_search,
            "q=needle&limit=not-an-int",
            "limit must be an integer",
        ),
        (
            "search count_max",
            handle_messages_search,
            "q=needle&count_max=not-an-int",
            "count_max must be an integer",
        ),
        (
            "tail limit",
            handle_messages_tail,
            "limit=not-an-int",
            "limit must be an integer",
        ),
        (
            "history limit",
            handle_messages_history,
            "cursor=dummy&limit=not-an-int",
            "limit must be an integer",
        ),
    ]
    for label, handler, query, expected_error in cases:
        deps, responses, _metrics = _deps()
        handler(
            _FakeHandler(),
            session_id="s1",
            query=query,
            manager=SimpleNamespace(
                refresh_session_meta=lambda _sid: None,
                get_session=lambda _sid: object(),
            ),
            deps=deps,
        )
        assert responses == [(400, {"error": expected_error})], label


if __name__ == "__main__":
    unittest.main()
