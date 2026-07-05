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
class ServerRouteCaps:
    ATTACH_UPLOAD_BODY_MAX_BYTES: Any
    ATTACH_UPLOAD_MAX_BYTES: Any
    CONTENT_SECURITY_POLICY: Any
    COOKIE_NAME: Any
    COOKIE_PATH: Any
    DEFAULT_AGENT_BACKEND: Any
    FILE_READ_MAX_BYTES: Any
    FILE_SEARCH_LIMIT: Any
    GIT_CHANGED_FILES_MAX: Any
    GIT_DIFF_MAX_BYTES: Any
    GIT_DIFF_TIMEOUT_SECONDS: Any
    LaunchRequestValidationError: Any
    MANAGER: Any
    STATIC_DIR: Any
    SessionCommitUnknownError: Any
    SessionInjectionError: Any
    SessionLaunchError: Any
    SessionNotReadyError: Any
    TMUX_SESSION_NAME: Any
    TOP_LEVEL_STATIC_ASSETS: Any
    TRANSCRIPT_EXPORT_MAX_BYTES: Any
    TRANSCRIPT_SEARCH_MAX_LINE_BYTES: Any
    _attachment_inject_text: Any
    _clean_unattended_cooldown_minutes: Any
    _clean_unattended_remaining_injections: Any
    _clip01: Any
    _current_git_branch: Any
    _decode_message_cursor: Any
    _describe_session_cwd: Any
    _download_disposition: Any
    _encode_message_cursor: Any
    _ensure_video_preview: Any
    _file_kind: Any
    _file_write_lock: Any
    _first_user_message_preview_from_log: Any
    _git_head_blob_oid: Any
    _inspect_downloadable_file: Any
    _is_same_password: Any
    _json_response: Any
    _json_response_with_etag: Any
    _launch_attempt_transcript_for_session_id: Any
    _list_resume_candidates_for_cwd: Any
    _list_session_relative_files: Any
    _list_session_relative_file_entries: Any
    _metrics_snapshot: Any
    _parse_git_numstat: Any
    _parse_new_session_launch_request: Any
    _provider_choice_for_settings: Any
    _read_body: Any
    _read_client_file_view: Any
    _read_new_session_defaults: Any
    _read_regular_file_prefix_no_symlink: Any
    _read_run_settings_from_log: Any
    _read_static_bytes: Any
    _read_text_file_strict: Any
    _record_metric: Any
    _require_auth: Any
    _require_git_repo: Any
    _resolve_client_file_path: Any
    _resolve_dir_target: Any
    _resolve_existing_absolute_file: Any
    _resolve_existing_session_file: Any
    _resolve_git_client_file_view: Any
    _resolve_git_existing_regular_file: Any
    _resolve_git_path: Any
    _resolve_session_cwd: Any
    _resolve_session_path: Any
    _resolve_under: Any
    _run_git: Any
    _search_session_relative_files: Any
    _select_runtime_token: Any
    _send_attachment_file_response: Any
    _send_inline_file_response: Any
    _set_auth_cookie: Any
    _sidebar_time_priority_from_elapsed_seconds: Any
    _split_git_nul_paths: Any
    _stage_uploaded_file: Any
    _static_asset_version: Any
    _static_cache_control_headers: Any
    _tmux_available: Any
    normalize_agent_backend: Any
    time: Any


def server_route_caps(server: Any) -> ServerRouteCaps:
    return ServerRouteCaps(
        ATTACH_UPLOAD_BODY_MAX_BYTES=server.ATTACH_UPLOAD_BODY_MAX_BYTES,
        ATTACH_UPLOAD_MAX_BYTES=server.ATTACH_UPLOAD_MAX_BYTES,
        CONTENT_SECURITY_POLICY=server.CONTENT_SECURITY_POLICY,
        COOKIE_NAME=server.COOKIE_NAME,
        COOKIE_PATH=server.COOKIE_PATH,
        DEFAULT_AGENT_BACKEND=server.DEFAULT_AGENT_BACKEND,
        FILE_READ_MAX_BYTES=server.FILE_READ_MAX_BYTES,
        FILE_SEARCH_LIMIT=server.FILE_SEARCH_LIMIT,
        GIT_CHANGED_FILES_MAX=server.GIT_CHANGED_FILES_MAX,
        GIT_DIFF_MAX_BYTES=server.GIT_DIFF_MAX_BYTES,
        GIT_DIFF_TIMEOUT_SECONDS=server.GIT_DIFF_TIMEOUT_SECONDS,
        LaunchRequestValidationError=server.LaunchRequestValidationError,
        MANAGER=server.MANAGER,
        STATIC_DIR=server.STATIC_DIR,
        SessionCommitUnknownError=server.SessionCommitUnknownError,
        SessionInjectionError=server.SessionInjectionError,
        SessionLaunchError=server.SessionLaunchError,
        SessionNotReadyError=server.SessionNotReadyError,
        TMUX_SESSION_NAME=server.TMUX_SESSION_NAME,
        TOP_LEVEL_STATIC_ASSETS=server.TOP_LEVEL_STATIC_ASSETS,
        TRANSCRIPT_EXPORT_MAX_BYTES=server.TRANSCRIPT_EXPORT_MAX_BYTES,
        TRANSCRIPT_SEARCH_MAX_LINE_BYTES=server.TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
        _attachment_inject_text=server._attachment_inject_text,
        _clean_unattended_cooldown_minutes=server._clean_unattended_cooldown_minutes,
        _clean_unattended_remaining_injections=server._clean_unattended_remaining_injections,
        _clip01=server._clip01,
        _current_git_branch=server._current_git_branch,
        _decode_message_cursor=server._decode_message_cursor,
        _describe_session_cwd=server._describe_session_cwd,
        _download_disposition=server._download_disposition,
        _encode_message_cursor=server._encode_message_cursor,
        _ensure_video_preview=server._ensure_video_preview,
        _file_kind=server._file_kind,
        _file_write_lock=server._file_write_lock,
        _first_user_message_preview_from_log=server._first_user_message_preview_from_log,
        _git_head_blob_oid=server._git_head_blob_oid,
        _inspect_downloadable_file=server._inspect_downloadable_file,
        _is_same_password=server._is_same_password,
        _json_response=server._json_response,
        _json_response_with_etag=server._json_response_with_etag,
        _launch_attempt_transcript_for_session_id=server._launch_attempt_transcript_for_session_id,
        _list_resume_candidates_for_cwd=server._list_resume_candidates_for_cwd,
        _list_session_relative_files=server._list_session_relative_files,
        _list_session_relative_file_entries=server._list_session_relative_file_entries,
        _metrics_snapshot=server._metrics_snapshot,
        _parse_git_numstat=server._parse_git_numstat,
        _parse_new_session_launch_request=server._parse_new_session_launch_request,
        _provider_choice_for_settings=server._provider_choice_for_settings,
        _read_body=server._read_body,
        _read_client_file_view=server._read_client_file_view,
        _read_new_session_defaults=server._read_new_session_defaults,
        _read_regular_file_prefix_no_symlink=server._read_regular_file_prefix_no_symlink,
        _read_run_settings_from_log=server._read_run_settings_from_log,
        _read_static_bytes=server._read_static_bytes,
        _read_text_file_strict=server._read_text_file_strict,
        _record_metric=server._record_metric,
        _require_auth=server._require_auth,
        _require_git_repo=server._require_git_repo,
        _resolve_client_file_path=server._resolve_client_file_path,
        _resolve_dir_target=server._resolve_dir_target,
        _resolve_existing_absolute_file=server._resolve_existing_absolute_file,
        _resolve_existing_session_file=server._resolve_existing_session_file,
        _resolve_git_client_file_view=server._resolve_git_client_file_view,
        _resolve_git_existing_regular_file=server._resolve_git_existing_regular_file,
        _resolve_git_path=server._resolve_git_path,
        _resolve_session_cwd=server._resolve_session_cwd,
        _resolve_session_path=server._resolve_session_path,
        _resolve_under=server._resolve_under,
        _run_git=server._run_git,
        _search_session_relative_files=server._search_session_relative_files,
        _select_runtime_token=server._select_runtime_token,
        _send_attachment_file_response=server._send_attachment_file_response,
        _send_inline_file_response=server._send_inline_file_response,
        _set_auth_cookie=server._set_auth_cookie,
        _sidebar_time_priority_from_elapsed_seconds=server._sidebar_time_priority_from_elapsed_seconds,
        _split_git_nul_paths=server._split_git_nul_paths,
        _stage_uploaded_file=server._stage_uploaded_file,
        _static_asset_version=server._static_asset_version,
        _static_cache_control_headers=server._static_cache_control_headers,
        _tmux_available=server._tmux_available,
        normalize_agent_backend=server.normalize_agent_backend,
        time=server.time,
    )


@dataclass(frozen=True)
class ServerRouteDepsFactory:
    caps: ServerRouteCaps

    def message_runtime_snapshot(
        self,
        session_id: str,
        session: Any,
        *,
        token_update: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, int, dict[str, Any] | None]:
        caps = self.caps
        manager = caps.MANAGER
        state = manager.get_state(session_id)
        log_available = session.log_path is not None and session.log_path.exists()
        runtime = manager._runtime_status_from_state_and_log(session_id, state, session.log_path)
        queue_val = manager._queue_len(session_id)
        token_val = caps._select_runtime_token(
            broker_state=state,
            session_token=session.token,
            token_update=token_update,
            log_available=log_available,
        )
        return state, bool(runtime.busy), int(queue_val), token_val

    def static_route_deps(self) -> StaticRouteDeps:
        caps = self.caps
        return StaticRouteDeps(
            static_dir=caps.STATIC_DIR,
            top_level_static_assets=caps.TOP_LEVEL_STATIC_ASSETS,
            read_static_bytes=caps._read_static_bytes,
            static_cache_control_headers=caps._static_cache_control_headers,
            content_security_policy=caps.CONTENT_SECURITY_POLICY,
        )

    def message_route_deps(self) -> MessageRouteDeps:
        caps = self.caps
        return MessageRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            launch_attempt_transcript_for_session_id=caps._launch_attempt_transcript_for_session_id,
            transcript_export_max_bytes=caps.TRANSCRIPT_EXPORT_MAX_BYTES,
            transcript_search_max_line_bytes=caps.TRANSCRIPT_SEARCH_MAX_LINE_BYTES,
            decode_message_cursor=caps._decode_message_cursor,
            encode_message_cursor=caps._encode_message_cursor,
            record_metric=caps._record_metric,
            message_runtime_snapshot=self.message_runtime_snapshot,
        )

    def queue_route_deps(self) -> QueueRouteDeps:
        caps = self.caps
        return QueueRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            read_json_body=lambda handler: handler._read_json_body(),
            session_not_ready_error=caps.SessionNotReadyError,
        )

    def hook_route_deps(self) -> HookRouteDeps:
        caps = self.caps
        return HookRouteDeps(
            read_body=caps._read_body,
            json_response=caps._json_response,
        )

    def control_route_deps(self) -> ControlRouteDeps:
        caps = self.caps
        return ControlRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            read_body=caps._read_body,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            attach_upload_body_max_bytes=caps.ATTACH_UPLOAD_BODY_MAX_BYTES,
            attach_upload_max_bytes=caps.ATTACH_UPLOAD_MAX_BYTES,
            stage_uploaded_file=caps._stage_uploaded_file,
            attachment_inject_text=caps._attachment_inject_text,
            clean_unattended_cooldown_minutes=caps._clean_unattended_cooldown_minutes,
            clean_unattended_remaining_injections=caps._clean_unattended_remaining_injections,
            session_not_ready_error=caps.SessionNotReadyError,
            session_injection_error=caps.SessionInjectionError,
            session_commit_unknown_error=caps.SessionCommitUnknownError,
        )

    def diagnostics_route_deps(self) -> DiagnosticsRouteDeps:
        caps = self.caps
        return DiagnosticsRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            provider_choice_for_settings=caps._provider_choice_for_settings,
            read_run_settings_from_log=caps._read_run_settings_from_log,
            resolve_session_cwd=caps._resolve_session_cwd,
            current_git_branch=caps._current_git_branch,
            sidebar_time_priority_from_elapsed_seconds=caps._sidebar_time_priority_from_elapsed_seconds,
            clip01=caps._clip01,
            time_fn=caps.time.time,
        )

    def auth_route_deps(self) -> AuthRouteDeps:
        caps = self.caps
        return AuthRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            is_same_password=caps._is_same_password,
            set_auth_cookie=caps._set_auth_cookie,
            cookie_name=caps.COOKIE_NAME,
            cookie_path=caps.COOKIE_PATH,
        )

    def session_route_deps(self) -> SessionRouteDeps:
        caps = self.caps
        return SessionRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            json_response_with_etag=caps._json_response_with_etag,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            read_new_session_defaults=caps._read_new_session_defaults,
            static_asset_version=caps._static_asset_version,
            tmux_available=caps._tmux_available,
            tmux_session_name=caps.TMUX_SESSION_NAME,
            metrics_snapshot=caps._metrics_snapshot,
            record_metric=caps._record_metric,
            perf_counter=caps.time.perf_counter,
            normalize_agent_backend=caps.normalize_agent_backend,
            default_agent_backend=caps.DEFAULT_AGENT_BACKEND,
            resolve_dir_target=caps._resolve_dir_target,
            describe_session_cwd=caps._describe_session_cwd,
            list_resume_candidates_for_cwd=caps._list_resume_candidates_for_cwd,
            first_user_message_preview_from_log=caps._first_user_message_preview_from_log,
            parse_new_session_launch_request=caps._parse_new_session_launch_request,
            launch_request_validation_error=caps.LaunchRequestValidationError,
            session_launch_error=caps.SessionLaunchError,
        )

    def voice_route_deps(self) -> VoiceRouteDeps:
        caps = self.caps
        return VoiceRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
        )

    def file_get_route_deps(self) -> FileGetRouteDeps:
        caps = self.caps
        return FileGetRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            resolve_session_cwd=caps._resolve_session_cwd,
            resolve_existing_session_file=caps._resolve_existing_session_file,
            resolve_session_path=caps._resolve_session_path,
            resolve_git_client_file_view=caps._resolve_git_client_file_view,
            resolve_git_existing_regular_file=caps._resolve_git_existing_regular_file,
            resolve_existing_absolute_file=caps._resolve_existing_absolute_file,
            read_client_file_view=caps._read_client_file_view,
            read_regular_file_prefix=caps._read_regular_file_prefix_no_symlink,
            search_session_relative_files=caps._search_session_relative_files,
            list_session_relative_files=caps._list_session_relative_files,
            list_session_relative_file_entries=caps._list_session_relative_file_entries,
            file_kind=caps._file_kind,
            ensure_video_preview=caps._ensure_video_preview,
            inspect_downloadable_file=caps._inspect_downloadable_file,
            download_disposition=caps._download_disposition,
            send_inline_file_response=caps._send_inline_file_response,
            send_attachment_file_response=caps._send_attachment_file_response,
            file_search_limit=caps.FILE_SEARCH_LIMIT,
        )

    def file_write_route_deps(self) -> FileWriteRouteDeps:
        caps = self.caps
        return FileWriteRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            resolve_session_cwd=caps._resolve_session_cwd,
            resolve_create_path=caps._resolve_under,
            resolve_git_existing_regular_file=caps._resolve_git_existing_regular_file,
            file_write_lock=caps._file_write_lock,
        )

    def global_file_route_deps(self) -> GlobalFileRouteDeps:
        caps = self.caps
        return GlobalFileRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            read_json_body=lambda handler, **kwargs: handler._read_json_body(**kwargs),
            resolve_git_client_file_view=caps._resolve_git_client_file_view,
            resolve_client_file_path=caps._resolve_client_file_path,
            read_client_file_view=caps._read_client_file_view,
        )

    def git_route_deps(self) -> GitRouteDeps:
        caps = self.caps
        return GitRouteDeps(
            require_auth=caps._require_auth,
            json_response=caps._json_response,
            resolve_session_cwd=caps._resolve_session_cwd,
            require_git_repo=caps._require_git_repo,
            split_git_nul_paths=caps._split_git_nul_paths,
            run_git=caps._run_git,
            parse_git_numstat=caps._parse_git_numstat,
            resolve_git_path=caps._resolve_git_path,
            read_text_file_strict=caps._read_text_file_strict,
            git_head_blob_oid=caps._git_head_blob_oid,
            git_changed_files_max=caps.GIT_CHANGED_FILES_MAX,
            git_diff_timeout_seconds=caps.GIT_DIFF_TIMEOUT_SECONDS,
            git_diff_max_bytes=caps.GIT_DIFF_MAX_BYTES,
            file_read_max_bytes=caps.FILE_READ_MAX_BYTES,
        )
