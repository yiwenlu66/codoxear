import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "codoxear" / "server.py"
FILE_UPLOAD_PY = ROOT / "codoxear" / "file_upload.py"


class TestFileUploadModuleSource(unittest.TestCase):
    def test_upload_helpers_live_outside_server_with_server_state_injection(self) -> None:
        server_source = SERVER_PY.read_text(encoding="utf-8")
        module_source = FILE_UPLOAD_PY.read_text(encoding="utf-8")

        self.assertIn("from .file_upload import safe_filename as _safe_filename", server_source)
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
        start = source.index('session_id = _match_session_route(path, "inject_file")')
        end = source.index('if path == "/api/hooks/notify":', start)
        block = source[start:end]
        self.assertIn("ready_for_attachment = MANAGER.attachment_injection_ready(session_id)", block)
        self.assertIn("resp = MANAGER.inject_attachment_keys(session_id, seq)", block)
        self.assertIn("self._refresh_session_meta_if_sidecar_exists(session_id, drain_queue=False)", source)
        self.assertIn("except SessionNotReadyError as e:", block)
        self.assertIn('{"error": "session is busy; wait before attaching a file"}', block)
        self.assertLess(block.index("ready_for_attachment = MANAGER.attachment_injection_ready(session_id)"), block.index("raw = base64.b64decode"))
        self.assertLess(block.index("ready_for_attachment = MANAGER.attachment_injection_ready(session_id)"), block.index("out_path = _stage_uploaded_file"))
        self.assertIn("with input_lock:\n            with self._lock:\n                s = self._sessions.get(session_id)", source)
        self.assertIn("if s.pending_attachment and not allow_pending_attachment:\n                    raise SessionNotReadyError", source)
        self.assertIn("if not self._send_remote_ready(session_id, allow_pending_attachment=allow_pending_attachment):\n                raise SessionNotReadyError(\"session is busy; wait before sending\")", source)
        self.assertIn("resp = self._sock_call(sock, {\"cmd\": \"send\", \"text\": text}, timeout_s=3.0)", source)
        self.assertIn("if s.pending_attachment:\n                    raise SessionNotReadyError(\"send the pending attachment before queueing another prompt\")", source)
        self.assertIn("self._record_prelog_user_message(s, text, source=\"enqueue\")\n            item, ql = self._queue_append_item_local(session_id, text)", source)
        self.assertIn("if isinstance(resp, dict) and resp.get(\"error\"):", source)
        self.assertIn("raise SessionInjectionError(str(resp.get(\"error\")))", source)
        self.assertIn("except SessionInjectionError as e:", block)
        self.assertIn("self._set_pending_attachment(session_id, True)", source)
        self.assertIn("self._set_pending_attachment(session_id, False)", source)
        self.assertIn("PENDING_ATTACHMENTS_PATH", source)
        self.assertIn("if s.pending_attachment and not allow_pending_attachment:", source)
        self.assertIn("if s.pending_attachment:\n                    raise SessionNotReadyError(\"send the pending attachment before queueing another prompt\")", source)
        self.assertIn('"pending_attachment": bool(s.pending_attachment)', source)


if __name__ == "__main__":
    unittest.main()
