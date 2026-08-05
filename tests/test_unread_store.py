import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from codoxear.session_routes import SessionRouteDeps
from codoxear.session_routes import _handle_read_post
from codoxear.session_routes import _handle_unread_get
from codoxear.unread_store import UnreadStore


class _Handler:
    pass


def _deps(payload, responses):
    return SessionRouteDeps(
        require_auth=lambda _handler: True,
        json_response=lambda _handler, status, body: responses.append((status, body)),
        json_response_with_etag=lambda *_args: None,
        read_json_body=lambda _handler: payload,
        read_new_session_defaults=lambda: {}, tmux_available=lambda: False,
        tmux_session_name="", metrics_snapshot=lambda: {}, record_metric=lambda *_args: None,
        perf_counter=lambda: 0.0, normalize_agent_backend=lambda value, default: default,
        default_agent_backend="pi", resolve_dir_target=lambda *_args, **_kwargs: Path("."),
        describe_session_cwd=lambda _path: {}, list_resume_candidates_for_cwd=lambda *_args, **_kwargs: [],
        first_user_message_preview_from_log=lambda _path: "", parse_new_session_launch_request=lambda _obj: None,
        launch_request_validation_error=ValueError, session_launch_error=RuntimeError,
    )


class TestUnreadStore(unittest.TestCase):
    def test_read_endpoint_advances_watermark_and_unread_returns_first_and_last(self):
        with tempfile.TemporaryDirectory() as td:
            store = UnreadStore(Path(td) / "session_unread.json")
            session = SimpleNamespace(log_path=Path(td) / "transcript.jsonl")
            manager = SimpleNamespace(_unread_store=store, get_session=lambda _sid: session)
            events = [
                {"role": "user", "text": "old", "message_id": "m1"},
                {"role": "assistant", "text": "old answer", "message_id": "m2"},
                {"role": "user", "text": "new", "message_id": "m3"},
                {"role": "assistant", "text": "new answer", "message_id": "m4"},
            ]
            with patch("codoxear.session_routes._transcript_events_for_unread", return_value=events):
                responses = []
                _handle_read_post(_Handler(), session_id="sid", manager=manager, deps=_deps({"event_id": "m2"}, responses))
                self.assertEqual(responses, [(200, {"ok": True, "event_id": "m2"})])
                responses.clear()
                _handle_unread_get(_Handler(), session_id="sid", manager=manager, deps=_deps({}, responses))
                self.assertEqual(responses, [(200, {"count": 2, "first_unread_event_id": "m3", "last_unread_event_id": "m4"})])

    def test_first_unread_initializes_existing_transcript_as_read(self):
        with tempfile.TemporaryDirectory() as td:
            store = UnreadStore(Path(td) / "session_unread.json")
            manager = SimpleNamespace(_unread_store=store, get_session=lambda _sid: SimpleNamespace(log_path=Path(td) / "x"))
            events = [{"role": "assistant", "text": "answer", "message_id": "m1"}]
            responses = []
            with patch("codoxear.session_routes._transcript_events_for_unread", return_value=events):
                _handle_unread_get(_Handler(), session_id="sid", manager=manager, deps=_deps({}, responses))
            self.assertEqual(responses[0][1]["count"], 0)
            self.assertEqual(store.watermark("sid"), "m1")

    def test_unread_endpoint_reads_bound_transcript_events(self):
        with tempfile.TemporaryDirectory() as td:
            transcript = Path(td) / "transcript.jsonl"
            transcript.write_text(
                '{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}],"phase":"final_answer"}}\n',
                encoding="utf-8",
            )
            store = UnreadStore(Path(td) / "session_unread.json")
            session = SimpleNamespace(log_path=transcript)
            manager = SimpleNamespace(_unread_store=store, get_session=lambda _sid: session)
            responses = []

            _handle_unread_get(_Handler(), session_id="sid", manager=manager, deps=_deps({}, responses))

            self.assertEqual(responses, [(200, {"count": 0, "first_unread_event_id": None, "last_unread_event_id": None})])
            self.assertTrue(store.watermark("sid"))


if __name__ == "__main__":
    unittest.main()
