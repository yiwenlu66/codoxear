from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .auth_routes import AuthRouteDeps
from .control_routes import ControlRouteDeps
from .diagnostics_routes import DiagnosticsRouteDeps
from .file_routes import FileGetRouteDeps
from .file_routes import FileWriteRouteDeps
from .file_routes import GlobalFileRouteDeps
from .git_routes import GitRouteDeps
from .hook_routes import HookRouteDeps
from .message_routes import MessageRouteDeps
from .queue_routes import QueueRouteDeps
from .session_routes import SessionRouteDeps
from .static_routes import StaticRouteDeps
from .voice_routes import VoiceRouteDeps


@dataclass(frozen=True)
class ServerRouteDepsFactory:
    server: Any

    def message_runtime_snapshot(
        self,
        session_id: str,
        session: Any,
        *,
        token_update: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, int, dict[str, Any] | None]:
        server = self.server
        manager = server.MANAGER
        state = manager.get_state(session_id)
        broker = server._runtime_broker_state(state)
        log_available = session.log_path is not None and session.log_path.exists()
        log_size = server._log_path_size_or_none(session.log_path)
        boundary_checker = getattr(manager, "_confirmed_send_boundary_unresolved_for_session", None)
        if callable(boundary_checker):
            boundary_unresolved = bool(boundary_checker(session_id, session.log_path, log_size))
        else:
            boundary_unresolved = server._consume_session_confirmed_send_boundary(session, session.log_path, log_size)
        log_idle = manager.idle_from_log(session_id) if log_available and not boundary_unresolved else None
        runtime = server._resolve_runtime_status(
            broker=broker,
            log_exists=log_available,
            log_idle=log_idle,
            send_boundary_unresolved=boundary_unresolved,
        )
        queue_val = manager._queue_len(session_id)
        token_val = server._select_runtime_token(
            broker_state=state,
            session_token=session.token,
            token_update=token_update,
            log_available=log_available,
        )
        return state, bool(runtime.busy), int(queue_val), token_val

    def static_route_deps(self) -> StaticRouteDeps:
        server = self.server
        return StaticRouteDeps(
            static_dir=server.STATIC_DIR,
            top_level_static_assets=server.TOP_LEVEL_STATIC_ASSETS,
            read_static_bytes=server._read_static_bytes,
            static_cache_control_headers=server._static_cache_control_headers,
            content_security_policy=server.CONTENT_SECURITY_POLICY,
        )

    def message_route_deps(self) -> MessageRouteDeps:
        server = self.server
        return MessageRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            launch_attempt_transcript_for_session_id=server._launch_attempt_transcript_for_session_id,
            transcript_export_max_bytes=server.TRANSCRIPT_EXPORT_MAX_BYTES,
            transcript_search_max_line_bytes=server.TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
            decode_message_cursor=server._decode_message_cursor,
            encode_message_cursor=server._encode_message_cursor,
            record_metric=server._record_metric,
            message_runtime_snapshot=self.message_runtime_snapshot,
        )

    def queue_route_deps(self) -> QueueRouteDeps:
        server = self.server
        return QueueRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            read_json_body=lambda handler: handler._read_json_body(),
            session_not_ready_error=server.SessionNotReadyError,
        )

    def hook_route_deps(self) -> HookRouteDeps:
        server = self.server
        return HookRouteDeps(
            read_body=server._read_body,
            json_response=server._json_response,
        )

    def control_route_deps(self) -> ControlRouteDeps:
        server = self.server
        return ControlRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            read_body=server._read_body,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            attach_upload_body_max_bytes=server.ATTACH_UPLOAD_BODY_MAX_BYTES,
            attach_upload_max_bytes=server.ATTACH_UPLOAD_MAX_BYTES,
            stage_uploaded_file=server._stage_uploaded_file,
            attachment_inject_text=server._attachment_inject_text,
            clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
            clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
            session_not_ready_error=server.SessionNotReadyError,
            session_injection_error=server.SessionInjectionError,
            session_commit_unknown_error=server.SessionCommitUnknownError,
        )

    def diagnostics_route_deps(self) -> DiagnosticsRouteDeps:
        server = self.server
        return DiagnosticsRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            provider_choice_for_settings=server._provider_choice_for_settings,
            read_run_settings_from_log=server._read_run_settings_from_log,
            resolve_session_cwd=server._resolve_session_cwd,
            current_git_branch=server._current_git_branch,
            sidebar_time_priority_from_elapsed_seconds=server._sidebar_time_priority_from_elapsed_seconds,
            clip01=server._clip01,
            time_fn=server.time.time,
        )

    def auth_route_deps(self) -> AuthRouteDeps:
        server = self.server
        return AuthRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            is_same_password=server._is_same_password,
            set_auth_cookie=server._set_auth_cookie,
            cookie_name=server.COOKIE_NAME,
            cookie_path=server.COOKIE_PATH,
        )

    def session_route_deps(self) -> SessionRouteDeps:
        server = self.server
        return SessionRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            json_response_with_etag=server._json_response_with_etag,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            read_new_session_defaults=server._read_new_session_defaults,
            static_asset_version=server._static_asset_version,
            tmux_available=server._tmux_available,
            tmux_session_name=server.TMUX_SESSION_NAME,
            metrics_snapshot=server._metrics_snapshot,
            record_metric=server._record_metric,
            perf_counter=server.time.perf_counter,
            normalize_agent_backend=server.normalize_agent_backend,
            default_agent_backend=server.DEFAULT_AGENT_BACKEND,
            resolve_dir_target=server._resolve_dir_target,
            describe_session_cwd=server._describe_session_cwd,
            list_resume_candidates_for_cwd=server._list_resume_candidates_for_cwd,
            first_user_message_preview_from_log=server._first_user_message_preview_from_log,
            parse_new_session_launch_request=server._parse_new_session_launch_request,
            launch_request_validation_error=server.LaunchRequestValidationError,
            session_launch_error=server.SessionLaunchError,
        )

    def voice_route_deps(self) -> VoiceRouteDeps:
        server = self.server
        return VoiceRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
        )

    def file_get_route_deps(self) -> FileGetRouteDeps:
        server = self.server
        return FileGetRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            resolve_session_cwd=server._resolve_session_cwd,
            resolve_existing_session_file=server._resolve_existing_session_file,
            resolve_session_path=server._resolve_session_path,
            resolve_git_client_file_view=server._resolve_git_client_file_view,
            resolve_git_existing_regular_file=server._resolve_git_existing_regular_file,
            resolve_existing_absolute_file=server._resolve_existing_absolute_file,
            read_client_file_view=server._read_client_file_view,
            search_session_relative_files=server._search_session_relative_files,
            list_session_relative_files=server._list_session_relative_files,
            file_kind=server._file_kind,
            ensure_video_preview=server._ensure_video_preview,
            inspect_downloadable_file=server._inspect_downloadable_file,
            download_disposition=server._download_disposition,
            send_inline_file_response=server._send_inline_file_response,
            send_attachment_file_response=server._send_attachment_file_response,
            file_search_limit=server.FILE_SEARCH_LIMIT,
        )

    def file_write_route_deps(self) -> FileWriteRouteDeps:
        server = self.server
        return FileWriteRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            resolve_session_cwd=server._resolve_session_cwd,
            resolve_create_path=server._resolve_under,
            resolve_git_existing_regular_file=server._resolve_git_existing_regular_file,
            file_write_lock=server._file_write_lock,
        )

    def global_file_route_deps(self) -> GlobalFileRouteDeps:
        server = self.server
        return GlobalFileRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            resolve_git_client_file_view=server._resolve_git_client_file_view,
            resolve_client_file_path=server._resolve_client_file_path,
            read_client_file_view=server._read_client_file_view,
        )

    def git_route_deps(self) -> GitRouteDeps:
        server = self.server
        return GitRouteDeps(
            require_auth=server._require_auth,
            json_response=server._json_response,
            resolve_session_cwd=server._resolve_session_cwd,
            require_git_repo=server._require_git_repo,
            split_git_nul_paths=server._split_git_nul_paths,
            run_git=server._run_git,
            parse_git_numstat=server._parse_git_numstat,
            resolve_git_path=server._resolve_git_path,
            read_text_file_strict=server._read_text_file_strict,
            git_head_blob_oid=server._git_head_blob_oid,
            git_changed_files_max=server.GIT_CHANGED_FILES_MAX,
            git_diff_timeout_seconds=server.GIT_DIFF_TIMEOUT_SECONDS,
            git_diff_max_bytes=server.GIT_DIFF_MAX_BYTES,
            file_read_max_bytes=server.FILE_READ_MAX_BYTES,
        )
