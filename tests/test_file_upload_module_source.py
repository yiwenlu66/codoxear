import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
SESSION_LISTING_PY = ROOT / "codoxear" / "session_listing.py"
SESSION_INPUT_PY = ROOT / "codoxear" / "session_input.py"
SESSION_CONTROL_PY = ROOT / "codoxear" / "session_control.py"
SESSION_READINESS_PY = ROOT / "codoxear" / "session_readiness.py"
SESSION_SEND_PY = ROOT / "codoxear" / "session_send.py"
SESSION_QUEUE_PY = ROOT / "codoxear" / "session_queue.py"
SESSION_ATTACHMENT_PY = ROOT / "codoxear" / "session_attachment.py"
SERVER_ROUTE_DEPS_PY = ROOT / "codoxear" / "server_route_deps.py"
FILE_UPLOAD_PY = ROOT / "codoxear" / "file_upload.py"
CONTROL_ROUTES_PY = ROOT / "codoxear" / "control_routes.py"
BROKER_PY = ROOT / "codoxear" / "broker.py"
SESSIOND_PY = ROOT / "codoxear" / "sessiond.py"
SESSION_LAUNCHER_PY = ROOT / "codoxear" / "session_launcher.py"


class TestFileUploadModuleSource(unittest.TestCase):
    def test_upload_helpers_live_outside_server_with_server_state_injection(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_UPLOAD_PY.read_text(encoding="utf-8")
        launcher_source = SESSION_LAUNCHER_PY.read_text(encoding="utf-8")

        self.assertNotIn("safe_filename as _safe_filename", server_source)
        self.assertIn("from .file_upload import safe_filename", launcher_source)
        self.assertIn("from .file_upload import stage_uploaded_file as _stage_uploaded_file_impl", server_source)
        self.assertIn("from .file_upload import attachment_inject_text as _attachment_inject_text", server_source)
        self.assertIn("upload_dir=UPLOAD_DIR", server_source)
        self.assertIn("now_fn=_now", server_source)
        self.assertNotIn("def _safe_filename(", server_source)
        self.assertNotIn("def _attachment_inject_text(", server_source)
        self.assertIn("def safe_filename(", module_source)
        self.assertIn("def stage_uploaded_file(", module_source)
        self.assertIn("def attachment_inject_text(", module_source)
        self.assertIn("upload_dir: Path", module_source)
        self.assertIn("now_fn: Callable[[], float]", module_source)

    def test_inject_file_route_checks_session_idle_before_staging(self) -> None:
        source = SERVER_PY.read_text(encoding="utf-8")
        listing_source = SESSION_LISTING_PY.read_text(encoding="utf-8")
        input_source = SESSION_INPUT_PY.read_text(encoding="utf-8")
        control_runtime_source = SESSION_CONTROL_PY.read_text(encoding="utf-8")
        readiness_source = SESSION_READINESS_PY.read_text(encoding="utf-8")
        send_source = SESSION_SEND_PY.read_text(encoding="utf-8")
        queue_source = SESSION_QUEUE_PY.read_text(encoding="utf-8")
        attachment_source = SESSION_ATTACHMENT_PY.read_text(encoding="utf-8")
        route_deps_source = SERVER_ROUTE_DEPS_PY.read_text(encoding="utf-8")
        route_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")
        start = route_source.index("def _handle_inject_attachment")
        block = route_source[start:]
        self.assertIn("if handle_control_post_route(", (ROOT / "codoxear" / "server_handler.py").read_text(encoding="utf-8"))
        self.assertIn("stage_uploaded_file=server._stage_uploaded_file", route_deps_source)
        self.assertIn("attachment_inject_text=server._attachment_inject_text", route_deps_source)
        self.assertIn("ready_for_attachment = manager.attachment_injection_ready(session_id)", block)
        self.assertIn("resp = manager.inject_attachment_keys(session_id, seq)", block)
        self.assertIn("self.refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)", readiness_source)
        self.assertIn("def refresh_session_meta(self, session_id: str, *, drain_queue: bool = False)", source)
        self.assertIn("def _refresh_session_meta_if_sidecar_exists(self, session_id: str, *, drain_queue: bool = False)", source)
        self.assertIn("except deps.session_not_ready_error as e:", block)
        self.assertIn('{"error": "session is busy; wait before attaching a file"}', block)
        self.assertLess(block.index("ready_for_attachment = manager.attachment_injection_ready(session_id)"), block.index("raw = base64.b64decode"))
        self.assertLess(block.index("ready_for_attachment = manager.attachment_injection_ready(session_id)"), block.index("out_path = deps.stage_uploaded_file"))
        self.assertIn("with input_lock:\n            with self.lock:\n                session = self.sessions().get(session_id)", send_source)
        self.assertIn("if session.pending_attachment and not allow_pending_attachment:\n        raise not_ready_error", input_source)
        self.assertIn("if not self.send_remote_ready(session_id, allow_pending_attachment=allow_pending_attachment):\n                raise self.not_ready_error(\"session is busy; wait before sending\")", send_source)
        self.assertIn("timeout_s = self.send_commit_timeout_seconds if self.send_commit_timeout_seconds > 0 else None", send_source)
        self.assertIn("response = self.call_confirmed_send", send_source)
        self.assertIn("{\"cmd\": \"send\", \"text\": text, \"sync\": True}", control_runtime_source)
        self.assertIn("except self.control_socket_call_error as exc:", control_runtime_source)
        self.assertIn("raise_commit_unknown(\"send commit status unknown; broker response failed\", exc)", control_runtime_source)
        self.assertIn("session_commit_unknown_error=caps.SessionCommitUnknownError", route_deps_source)
        self.assertIn("except deps.session_commit_unknown_error as e:", route_source)
        self.assertIn('"commit_unknown": True', route_source + source)
        self.assertIn("if bool(response.get(\"commit_unknown\")):\n        raise_commit_unknown(\"send commit status unknown; broker marked commit unknown\")", input_source)
        self.assertIn("if bool(response.get(\"commit_unknown\")):\n                self.set_pending_attachment(session_id, True)\n                raise self.commit_unknown_error(\"attachment commit status unknown; broker marked commit unknown\")", attachment_source)
        self.assertIn('(\"pending_attachment\", \"clear\", _handle_pending_attachment_clear)', route_source)
        self.assertIn("res = manager.clear_pending_attachment(session_id)", route_source)
        self.assertIn("if not session.sync_send_supported:", input_source)
        self.assertIn("if not (session.sync_send_supported and session.key_write_errors_supported):", readiness_source)
        self.assertIn("response = self.inject_keys(session_id, seq, track_request_sent=True)", attachment_source)
        self.assertIn("def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)", source)
        self.assertIn("attachment commit status unknown; broker response failed", control_runtime_source)
        self.assertIn("except deps.session_not_ready_error as e:", block)
        self.assertIn("if session.pending_attachment:\n                    raise self.not_ready_error(\"send the pending attachment before queueing another prompt\")", queue_source)
        self.assertNotIn("self._record_prelog_user_message(s, text, source=\"enqueue\")", source)
        self.assertIn("if response.get(\"error\"):", attachment_source)
        self.assertIn("if response.get(\"ok\") is not True:", attachment_source)
        self.assertIn("raise self.injection_error(err)", attachment_source)
        self.assertIn("except deps.session_injection_error as e:", block)
        self.assertIn("self.set_pending_attachment(session_id, True)", attachment_source)
        self.assertIn("self.set_pending_attachment(session_id, False)", send_source)
        self.assertIn("PENDING_ATTACHMENTS_PATH", source)
        self.assertIn("if session.pending_attachment and not allow_pending_attachment:", input_source)
        self.assertIn("if session.pending_attachment:\n                    raise self.not_ready_error(\"send the pending attachment before queueing another prompt\")", queue_source)
        self.assertIn('pending_attachment=bool(s.pending_attachment)', listing_source)
        self.assertIn('commit_unknown_send=s.commit_unknown_send if isinstance(s.commit_unknown_send, dict) else None', listing_source)
        self.assertIn('(\"commit_unknown_send\", \"clear\", _handle_commit_unknown_send_clear)', route_source)
        self.assertIn("res = manager.clear_commit_unknown_send(session_id)", route_source)
        self.assertIn("if session.commit_unknown_send:\n        raise not_ready_error(\"resolve the unknown send before submitting more text\")", input_source)
        self.assertIn("if session.commit_unknown_send:\n                raise self.not_ready_error(\"resolve the unknown send before attaching a file\")", readiness_source)
        self.assertIn("if session.commit_unknown_send:\n                    raise self.not_ready_error(\"resolve the unknown send before queueing another prompt\")", queue_source)

    def test_control_sidecars_advertise_sync_send_capability(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        sessiond_source = SESSIOND_PY.read_text(encoding="utf-8")

        for source in [broker_source, sessiond_source]:
            self.assertIn('"control_protocol_version": 2', source)
            self.assertIn('"control_capabilities": {"sync_send": True, "key_write_errors": True}', source)


if __name__ == "__main__":
    unittest.main()
