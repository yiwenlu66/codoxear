import json
import types
import unittest
import urllib.parse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import server
from codoxear.server import _casefold_match_span
from codoxear.server import _clip_search_match_text
from codoxear.server import _clip_search_text_around_query
from codoxear.server import _read_chat_export_events
from codoxear.server import _search_chat_events
from codoxear.server import _search_chat_log
from codoxear.server import Session


APP_JS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.js"
SERVER_PY = Path(__file__).resolve().parents[1] / "codoxear" / "server.py"


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

    def test_streaming_search_counts_logs_that_export_rejects(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "rollout.jsonl"
            _write_assistant_rows(path, 25)
            with self.assertRaisesRegex(ValueError, "too large to export"):
                _read_chat_export_events(path, max_bytes=1)

            count, matches = _search_chat_log(path, "a2", limit=3, max_line_bytes=4096)

        self.assertEqual(count, 6)
        self.assertEqual([match.get("text") for match in matches], ["a2", "a20", "a21"])
        self.assertTrue(all(isinstance(match.get("_before_byte"), int) for match in matches))

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

            count, matches = _search_chat_log(path, "needle", limit=10, max_line_bytes=4096)

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

            count, matches = _search_chat_log(path, "needle", limit=5, max_line_bytes=4096)

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

            count, matches = _search_chat_log(path, "needle", limit=5, max_line_bytes=4096)

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

            count, matches = _search_chat_log(path, "needle", limit=5, max_line_bytes=10000)

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

            count, matches = _search_chat_log(path, "needle", limit=5, max_line_bytes=10000)

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

            count, matches = _search_chat_log(path, "needle", limit=5, max_line_bytes=1024)

        self.assertEqual(count, 1)
        self.assertEqual(matches[0].get("text"), "needle after oversized")

    def test_search_match_text_clipping_is_opt_in_and_query_centered(self) -> None:
        matches = [
            {"role": "assistant", "text": "x" * 40 + "needle" + "y" * 40, "_before_byte": 1},
            {"role": "user", "text": "xy", "_before_byte": 2},
        ]

        self.assertIs(_clip_search_match_text(matches, 0, query="needle"), matches)
        clipped = _clip_search_match_text(matches, 18, query="needle")

        self.assertLessEqual(len(clipped[0]["text"]), 18)
        self.assertIn("needle", clipped[0]["text"])
        self.assertFalse(clipped[0]["text"].startswith("…"))
        self.assertFalse(clipped[0]["text"].endswith("…"))
        self.assertTrue(clipped[0]["text_truncated"])
        self.assertEqual(clipped[0]["_before_byte"], 1)
        self.assertEqual(clipped[1]["text"], "xy")
        self.assertNotIn("text_truncated", clipped[1])
        self.assertEqual(matches[0]["text"], "x" * 40 + "needle" + "y" * 40)
        prefix, truncated = _clip_search_text_around_query("abcdef", "missing", 3)
        self.assertEqual(prefix, "abc")
        self.assertTrue(truncated)

    def test_search_text_clipping_maps_casefold_offsets_to_original_text(self) -> None:
        self.assertEqual(_casefold_match_span("ß" * 4 + "needle", "needle"), (4, 10))
        snippet, truncated = _clip_search_text_around_query("ß" * 40 + "needle" + "y" * 40, "needle", 18)

        self.assertTrue(truncated)
        self.assertIn("needle", snippet)
        self.assertNotEqual(snippet, "y" * 18)
        exact, exact_truncated = _clip_search_text_around_query("xxxneedlezzz", "needle", 6)
        self.assertEqual(exact, "needle")
        self.assertTrue(exact_truncated)

    def test_search_counts_all_matching_chat_events(self) -> None:
        events = [
            {"role": "user", "text": "Needle in first user turn", "_before_byte": 1},
            {"role": "assistant", "text": "no match", "_before_byte": 2},
            {"role": "assistant", "text": "another NEEDLE appears", "_before_byte": 3},
            {"role": "system", "text": "needle ignored", "_before_byte": 4},
        ]

        count, matches = _search_chat_events(events, "needle", limit=1)

        self.assertEqual(count, 2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["_before_byte"], 1)

    def test_server_exposes_messages_export_and_search_routes(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertIn('_match_session_route(path, "messages", "export")', source)
        self.assertIn('_match_session_route(path, "messages", "search")', source)
        self.assertIn('"event_count": len(events)', source)
        self.assertIn('"match_count": match_count', source)
        self.assertIn('match_count, matches = _search_chat_log(s.log_path, query, limit=match_limit)', source)
        self.assertIn('text_max, text_max_error = _parse_bounded_query_int(qs, "text_max", default=0, min_value=0, max_value=4096)', source)
        self.assertIn('matches = _clip_search_match_text(matches, text_max, query=query)', source)
        self.assertIn('TRANSCRIPT_SEARCH_MAX_LINE_BYTES', source)
        self.assertIn('def _parse_bounded_query_int(', source)
        self.assertIn('_json_response(self, 400, {"error": limit_error})', source)
        self.assertIn('_json_response(self, 413', source)

    def test_messages_search_route_applies_text_max_to_response_matches(self) -> None:
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
            session = Session(
                session_id="s1",
                thread_id="thread-1",
                broker_pid=1,
                codex_pid=1,
                agent_backend="codex",
                owned=False,
                start_ts=0.0,
                cwd=str(Path(td)),
                log_path=log_path,
                sock_path=Path(td) / "s1.sock",
            )
            fake_manager = types.SimpleNamespace(
                refresh_session_meta=lambda _sid: None,
                get_session=lambda _sid: session,
                _attach_notification_texts=lambda matches: matches,
            )
            responses = []
            handler = server.Handler.__new__(server.Handler)
            parsed = urllib.parse.urlparse("/api/sessions/s1/messages/search?q=needle&limit=1&text_max=18")
            handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
            handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]

            with patch.object(server, "MANAGER", fake_manager), patch.object(server, "_require_auth", return_value=True), patch.object(
                server,
                "_json_response",
                side_effect=lambda _handler, status, obj: responses.append((status, obj)),
            ):
                server.Handler.do_GET(handler)

            self.assertEqual(len(responses), 1)
            status, body = responses[0]
            self.assertEqual(status, 200)
            self.assertEqual(body["transcript_state"], "bound")
            self.assertEqual(body["thread_id"], "thread-1")
            self.assertEqual(body["log_path"], str(log_path))
            self.assertEqual(body["match_count"], 2)
            self.assertEqual(len(body["matches"]), 1)
            match = body["matches"][0]
            self.assertLessEqual(len(match["text"]), 18)
            self.assertIn("needle", match["text"])
            self.assertTrue(match["text_truncated"])
            self.assertIsInstance(match.get("_before_byte"), int)

    def test_messages_search_rejects_malformed_text_max(self) -> None:
        fake_manager = types.SimpleNamespace(
            refresh_session_meta=lambda _sid: None,
            get_session=lambda _sid: object(),
        )
        responses = []
        handler = server.Handler.__new__(server.Handler)
        parsed = urllib.parse.urlparse("/api/sessions/s1/messages/search?q=needle&text_max=bad")
        handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
        handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]

        with patch.object(server, "MANAGER", fake_manager), patch.object(server, "_require_auth", return_value=True), patch.object(
            server,
            "_json_response",
            side_effect=lambda _handler, status, obj: responses.append((status, obj)),
        ):
            server.Handler.do_GET(handler)

        self.assertEqual(responses, [(400, {"error": "text_max must be an integer"})])

    def test_message_routes_reject_malformed_limits(self) -> None:
        fake_manager = types.SimpleNamespace(
            refresh_session_meta=lambda _sid: None,
            get_session=lambda _sid: object(),
        )
        paths = [
            "/api/sessions/s1/messages/search?q=needle&limit=not-an-int",
            "/api/sessions/s1/messages/tail?limit=not-an-int",
            "/api/sessions/s1/messages/history?cursor=dummy&limit=not-an-int",
        ]
        for path in paths:
            with self.subTest(path=path):
                responses = []
                handler = server.Handler.__new__(server.Handler)
                parsed = urllib.parse.urlparse(path)
                handler._parse_prefixed_request_path = lambda parsed=parsed: (parsed, parsed.path)  # type: ignore[attr-defined]
                handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]

                with patch.object(server, "MANAGER", fake_manager), patch.object(server, "_require_auth", return_value=True), patch.object(
                    server,
                    "_json_response",
                    side_effect=lambda _handler, status, obj: responses.append((status, obj)),
                ):
                    server.Handler.do_GET(handler)

                self.assertEqual(responses, [(400, {"error": "limit must be an integer"})])

    def test_ui_has_copy_conversation_action(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertIn('id: "copyConversationBtn"', source)
        self.assertIn('function formatConversationForCopy(events)', source)
        self.assertIn('api(`/api/sessions/${sid}/messages/export`)', source)
        self.assertIn('title: "Copy conversation"', source)


if __name__ == "__main__":
    unittest.main()
