import json
import types
import unittest
import urllib.parse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codoxear import server
from codoxear.server import _read_chat_export_events
from codoxear.server import _search_chat_events


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
        self.assertIn('_json_response(self, 400, {"error": "limit must be an integer"})', source)
        self.assertIn('_json_response(self, 413', source)

    def test_messages_search_rejects_malformed_limit(self) -> None:
        responses = []
        handler = server.Handler.__new__(server.Handler)
        parsed = urllib.parse.urlparse("/api/sessions/s1/messages/search?q=needle&limit=not-an-int")
        handler._parse_prefixed_request_path = lambda: (parsed, parsed.path)  # type: ignore[attr-defined]
        handler._handle_static_get = lambda _path: False  # type: ignore[attr-defined]
        fake_manager = types.SimpleNamespace(
            refresh_session_meta=lambda _sid: None,
            get_session=lambda _sid: object(),
        )

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
