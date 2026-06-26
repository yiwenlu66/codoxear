import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
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
        route_source = CONTROL_ROUTES_PY.read_text(encoding="utf-8")
        start = route_source.index("def _handle_inject_attachment")
        block = route_source[start:]
        self.assertIn("_handle_control_post_route(", source)
        self.assertIn("stage_uploaded_file=_stage_uploaded_file", source)
        self.assertIn("attachment_inject_text=_attachment_inject_text", source)
        self.assertIn("ready_for_attachment = manager.attachment_injection_ready(session_id)", block)
        self.assertIn("resp = manager.inject_attachment_keys(session_id, seq)", block)
        self.assertIn("self._refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)", source)
        self.assertIn("def refresh_session_meta(self, session_id: str, *, drain_queue: bool = False)", source)
        self.assertIn("def _refresh_session_meta_if_sidecar_exists(self, session_id: str, *, drain_queue: bool = False)", source)
        self.assertIn("except deps.session_not_ready_error as e:", block)
        self.assertIn('{"error": "session is busy; wait before attaching a file"}', block)
        self.assertLess(block.index("ready_for_attachment = manager.attachment_injection_ready(session_id)"), block.index("raw = base64.b64decode"))
        self.assertLess(block.index("ready_for_attachment = manager.attachment_injection_ready(session_id)"), block.index("out_path = deps.stage_uploaded_file"))
        self.assertIn("with input_lock:\n            with self._lock:\n                s = self._sessions.get(session_id)", source)
        self.assertIn("if s.pending_attachment and not allow_pending_attachment:\n                    raise SessionNotReadyError", source)
        self.assertIn("if not self._send_remote_ready(session_id, allow_pending_attachment=allow_pending_attachment):\n                raise SessionNotReadyError(\"session is busy; wait before sending\")", source)
        self.assertIn("timeout_s = SEND_COMMIT_TIMEOUT_SECONDS if SEND_COMMIT_TIMEOUT_SECONDS > 0 else None", source)
        self.assertIn("resp = self._sock_call(sock, {\"cmd\": \"send\", \"text\": text, \"sync\": True}, timeout_s=timeout_s, track_request_sent=True)", source)
        self.assertIn("except ControlSocketCallError as e:", source)
        self.assertIn("if e.request_sent:\n                    raise_commit_unknown(\"send commit status unknown; broker response failed\", e)", source)
        self.assertIn("session_commit_unknown_error=SessionCommitUnknownError", source)
        self.assertIn("except deps.session_commit_unknown_error as e:", route_source)
        self.assertIn('"commit_unknown": True', route_source + source)
        self.assertIn("if bool(resp.get(\"commit_unknown\")):\n                raise_commit_unknown(\"send commit status unknown; broker marked commit unknown\")", source)
        self.assertIn("if bool(resp.get(\"commit_unknown\")):\n                self._set_pending_attachment(session_id, True)\n                raise SessionCommitUnknownError(\"attachment commit status unknown; broker marked commit unknown\")", source)
        self.assertIn('(\"pending_attachment\", \"clear\", _handle_pending_attachment_clear)', route_source)
        self.assertIn("res = manager.clear_pending_attachment(session_id)", route_source)
        self.assertIn("if not s.sync_send_supported:", source)
        self.assertIn("if not (s.sync_send_supported and s.key_write_errors_supported):", source)
        self.assertIn("resp = self.inject_keys(session_id, seq, track_request_sent=True)", source)
        self.assertIn("def inject_keys(self, session_id: str, seq: str, *, track_request_sent: bool = False, interrupt: bool = False)", source)
        self.assertIn("attachment commit status unknown; broker response failed", source)
        self.assertIn("except deps.session_not_ready_error as e:", block)
        self.assertIn("if s.pending_attachment:\n                    raise SessionNotReadyError(\"send the pending attachment before queueing another prompt\")", source)
        self.assertNotIn("self._record_prelog_user_message(s, text, source=\"enqueue\")", source)
        self.assertIn("if resp.get(\"error\"):", source)
        self.assertIn("if resp.get(\"ok\") is not True:", source)
        self.assertIn("raise SessionInjectionError(err)", source)
        self.assertIn("except deps.session_injection_error as e:", block)
        self.assertIn("self._set_pending_attachment(session_id, True)", source)
        self.assertIn("self._set_pending_attachment(session_id, False)", source)
        self.assertIn("PENDING_ATTACHMENTS_PATH", source)
        self.assertIn("if s.pending_attachment and not allow_pending_attachment:", source)
        self.assertIn("if s.pending_attachment:\n                    raise SessionNotReadyError(\"send the pending attachment before queueing another prompt\")", source)
        self.assertIn('"pending_attachment": bool(s.pending_attachment)', source)
        self.assertIn('"commit_unknown_send": bool(s.commit_unknown_send)', source)
        self.assertIn('(\"commit_unknown_send\", \"clear\", _handle_commit_unknown_send_clear)', route_source)
        self.assertIn("res = manager.clear_commit_unknown_send(session_id)", route_source)
        self.assertIn("if s.commit_unknown_send:\n                    raise SessionNotReadyError(\"resolve the unknown send before submitting more text\")", source)
        self.assertIn("if s.commit_unknown_send:\n                raise SessionNotReadyError(\"resolve the unknown send before attaching a file\")", source)
        self.assertIn("if s.commit_unknown_send:\n                    raise SessionNotReadyError(\"resolve the unknown send before queueing another prompt\")", source)

    def test_control_sidecars_advertise_sync_send_capability(self) -> None:
        broker_source = BROKER_PY.read_text(encoding="utf-8")
        sessiond_source = SESSIOND_PY.read_text(encoding="utf-8")

        for source in [broker_source, sessiond_source]:
            self.assertIn('"control_protocol_version": 2', source)
            self.assertIn('"control_capabilities": {"sync_send": True, "key_write_errors": True}', source)


if __name__ == "__main__":
    unittest.main()
