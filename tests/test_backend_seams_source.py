import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "codoxear"
SERVER = ROOT / "server.py"
VOICE_PUSH = ROOT / "voice_push.py"


class TestBackendSeamsSource(unittest.TestCase):
    def test_server_dispatches_http_routes_through_modules(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .http.routes import assets as _http_assets_routes", source)
        self.assertIn("from .http.routes import sessions_read as _http_session_read_routes", source)
        self.assertIn("route_module.handle_get(RUNTIME, self, path, u)", source)
        self.assertIn("route_module.handle_post(RUNTIME, self, path, u)", source)

    def test_server_uses_payload_sidebar_and_pi_bridge_seams(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .sessions import payloads as _session_payloads", source)
        self.assertIn("from .sessions import live_payloads as _session_live_payloads", source)
        self.assertIn("from .sessions import pi_session_files as _pi_session_files", source)
        self.assertIn("from .sessions import resume_candidates as _resume_candidates", source)
        self.assertIn("from .sessions import background as _session_background", source)
        self.assertIn("from .sessions import lifecycle as _session_lifecycle", source)
        self.assertIn("from .sessions import page_state as _page_state", source)
        self.assertIn("from .sessions import listing as _session_listing", source)
        self.assertIn("from .sessions import session_catalog as _session_catalog", source)
        self.assertIn("from .sessions import session_control as _session_control", source)
        self.assertIn("from .sessions import sidebar_state as _sidebar_state_module", source)
        self.assertIn("from .sessions import transport as _session_transport", source)
        self.assertIn("from .workspace import file_access as _workspace_file_access", source)
        self.assertIn("from .workspace import file_search as _workspace_file_search", source)
        self.assertIn("from .pi import ui_bridge as _pi_ui_bridge", source)
        self.assertIn("self._sidebar_state_facade().persist_session_ui_state()", source)
        self.assertIn("return _pi_ui_bridge.submit_ui_response(RUNTIME, self, session_id, payload)", source)

    def test_server_builds_explicit_runtime_object(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .runtime import ServerRuntime, build_server_runtime", source)
        self.assertIn("RUNTIME = MANAGER._runtime", source)
        self.assertIn("build_server_runtime(", source)

    def test_file_routes_delegate_workspace_behavior_to_service_owner(self) -> None:
        source = (ROOT / "http" / "routes" / "files.py").read_text(encoding="utf-8")
        self.assertIn("from ...workspace import service as _workspace_service", source)
        self.assertIn("_workspace_service.read_session_file", source)
        self.assertIn("_workspace_service.search_session_files", source)
        self.assertIn("_workspace_service.list_session_files", source)
        self.assertIn("_workspace_service.write_session_file", source)
        self.assertIn("_workspace_service.inject_session_attachment", source)

    def test_session_routes_delegate_creation_to_owner_and_keep_server_listing_surfaces(self) -> None:
        read_source = (ROOT / "http" / "routes" / "sessions_read.py").read_text(encoding="utf-8")
        write_source = (ROOT / "http" / "routes" / "sessions_write.py").read_text(encoding="utf-8")
        self.assertIn("from ...sessions import creation as _session_creation", read_source)
        self.assertIn("_session_creation.read_new_session_defaults(", read_source)
        self.assertIn("sv._session_list_payload(", read_source)
        self.assertIn("sv._first_user_message_preview_from_log", read_source)
        self.assertIn("sv._first_user_message_preview_from_pi_session", read_source)
        self.assertIn("from ...sessions import creation as _session_creation", write_source)
        self.assertIn("_session_creation.parse_create_session_request(sv, obj)", write_source)

    def test_server_delegates_session_catalog_identity_lookups(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .sessions import session_catalog as _session_catalog", source)
        self.assertIn("return _session_catalog.service(self).runtime_session_id_for_identifier(session_id)", source)
        self.assertIn("return _session_catalog.service(self).durable_session_id_for_identifier(session_id)", source)
        self.assertIn("return _session_catalog.service(self).page_state_ref_for_session_id(session_id)", source)
        self.assertIn("return _session_catalog.service(self).get_session(session_id)", source)
        self.assertIn("return _session_catalog.service(self).list_sessions()", source)
        self.assertIn("_session_catalog.service(self).refresh_session_meta(session_id, strict=strict)", source)
        self.assertIn("_session_catalog.service(self).discover_existing(", source)
        self.assertIn("return _session_catalog.service(self).refresh_session_state(", source)
        self.assertIn("_session_catalog.service(self).prune_dead_sessions()", source)

    def test_server_delegates_session_control_send_and_queue_flows(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("from .sessions import session_control as _session_control", source)
        self.assertIn("return _session_control.service(self).send(session_id, text)", source)
        self.assertIn("return _session_control.service(self).enqueue(session_id, text)", source)
        self.assertIn("return _session_control.service(self).queue_list(session_id)", source)
        self.assertIn("return _session_control.service(self).queue_delete(session_id, int(index))", source)
        self.assertIn("return _session_control.service(self).queue_update(session_id, int(index), text)", source)
        self.assertIn("return _session_control.service(self).restart_session(session_id)", source)
        self.assertIn("return _session_control.service(self).handoff_session(session_id)", source)
        self.assertIn("return _session_control.service(self).spawn_web_session(", source)
        self.assertIn("return _message_history.service(self).get_messages_page(", source)
        self.assertIn("return _message_history.service(self).ensure_chat_index(", source)
        self.assertIn("return _message_history.service(self).ensure_pi_chat_index(", source)
        self.assertIn("_message_history.service(self).mark_log_delta(", source)
        self.assertIn("return _message_history.service(self).idle_from_log(session_id)", source)
        self.assertIn("return _page_state.service(self).queue_len(session_id)", source)
        self.assertIn("return _page_state.service(self).queue_enqueue_local(session_id, text)", source)
        self.assertIn("return _page_state.service(self).files_get(session_id)", source)
        self.assertIn("return _page_state.service(self).harness_get(session_id)", source)
        self.assertIn("return _page_state.service(self).cwd_group_set(", source)
        self.assertIn("return _page_state.service(self).recent_cwds(limit=limit)", source)
        self.assertIn("return _session_background.service(self).probe_bridge_transport(", source)
        self.assertIn("return _session_background.service(self).enqueue_outbound_request(runtime_id, text)", source)
        self.assertIn("return _session_background.service(self).maybe_drain_outbound_request(runtime_id)", source)
        self.assertIn("return _session_background.service(self).session_display_name(session_id)", source)
        self.assertIn("_session_background.service(self).observe_rollout_delta(", source)
        self.assertIn("_session_background.service(self).harness_sweep()", source)
        self.assertIn("_session_background.service(self).queue_sweep()", source)
        self.assertIn("_session_background.service(self).update_meta_counters()", source)
        self.assertIn("return _session_lifecycle.service(self).catalog_record_for_ref(ref)", source)
        self.assertIn("_session_lifecycle.service(self).refresh_durable_session_catalog(force=force)", source)
        self.assertIn("return _session_lifecycle.service(self).wait_for_live_session(", source)
        self.assertIn("return _session_lifecycle.service(self).capture_runtime_bound_restart_state(", source)
        self.assertIn("_session_lifecycle.service(self).stage_runtime_bound_restart_state(", source)
        self.assertIn("_session_lifecycle.service(self).restore_runtime_bound_restart_state(", source)
        self.assertIn("_session_lifecycle.service(self).finalize_pending_pi_spawn(", source)
        self.assertIn("_session_lifecycle.service(self).reset_log_caches(s, meta_log_off=meta_log_off)", source)
        self.assertIn("return _session_lifecycle.service(self).claimed_pi_session_paths(", source)
        self.assertIn("_session_lifecycle.service(self).apply_session_source(", source)
        self.assertIn("return _session_lifecycle.service(self).session_run_settings(", source)
        self.assertIn("return _session_transport.service(self).get_state(session_id)", source)
        self.assertIn("return _session_transport.service(self).get_tail(session_id)", source)
        self.assertIn("return _session_transport.service(self).inject_keys(session_id, seq)", source)
        self.assertIn("return _session_transport.service(self).kill_session(session_id)", source)
        self.assertIn("return _workspace_file_access.resolve_client_file_path(", source)
        self.assertIn("return _workspace_file_access.read_client_file_view(RUNTIME, path_obj)", source)
        self.assertIn("return _workspace_file_access.read_downloadable_file(path_obj)", source)
        self.assertIn("return _workspace_file_access.download_disposition(path_obj)", source)
        self.assertIn("return _workspace_file_search.search_session_relative_files(", source)
        self.assertIn("return _resume_candidates.resume_candidate_from_log(", source)
        self.assertIn("return _resume_candidates.resolve_pi_session_path(", source)
        self.assertIn("return _resume_candidates.list_resume_candidates_for_cwd(", source)
        self.assertIn("return _resume_candidates.iter_all_resume_candidates(RUNTIME, limit=limit)", source)
        self.assertIn("return _resume_candidates.pi_resume_candidate_from_session_file(", source)
        self.assertIn("return _session_listing.historical_session_id(RUNTIME, backend, resume_session_id)", source)
        self.assertIn("return _session_listing.parse_historical_session_id(RUNTIME, session_id)", source)
        self.assertIn("return _session_listing.historical_session_row(RUNTIME, session_id)", source)
        self.assertIn("return _pi_session_files.pi_new_session_file_for_cwd(RUNTIME, cwd)", source)
        self.assertIn("return _pi_session_files.write_pi_session_header(", source)
        self.assertIn("return _pi_session_files.next_pi_handoff_history_path(session_path)", source)
        self.assertIn("return _pi_session_files.copy_file_atomic(RUNTIME, source_path, target_path)", source)
        self.assertIn("return _pi_session_files.write_pi_handoff_session(", source)
        self.assertIn("return _pi_session_files.pi_session_has_handoff_history(session_path)", source)
        self.assertIn("return _pi_session_files.pi_session_name_from_session_file(", source)

    def test_seam_modules_no_longer_use_bound_server_globals(self) -> None:
        files = [
            ROOT / "http" / "routes" / "assets.py",
            ROOT / "http" / "routes" / "auth.py",
            ROOT / "http" / "routes" / "events.py",
            ROOT / "http" / "routes" / "files.py",
            ROOT / "http" / "routes" / "notifications.py",
            ROOT / "http" / "routes" / "sessions_read.py",
            ROOT / "http" / "routes" / "sessions_write.py",
            ROOT / "sessions" / "payloads.py",
            ROOT / "sessions" / "live_payloads.py",
            ROOT / "pi" / "ui_bridge.py",
        ]
        for path in files:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("_SERVER = None", source)
            self.assertNotIn("bind_server_runtime", source)
            self.assertNotIn("def _sv(", source)

    def test_server_runtime_exposes_pi_context_window_helper(self) -> None:
        source = SERVER.read_text(encoding="utf-8")
        self.assertIn("_pi_model_context_window_impl", source)

    def test_voice_push_uses_attention_namespace(self) -> None:
        source = VOICE_PUSH.read_text(encoding="utf-8")
        self.assertIn("from .attention.derive import", source)
        self.assertIn("compact_notification_state", source)
        self.assertIn("final_response_attention_feed", source)
        self.assertIn("return final_response_attention_feed(", source)
        self.assertNotIn("from codoxear import server as sv", source)


if __name__ == "__main__":
    unittest.main()
