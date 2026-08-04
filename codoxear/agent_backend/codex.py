from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from ..subagent_events import emit_subagent_event
from .base import AgentBackend, _SESSION_ID_RE


class CodexBackend(AgentBackend):
    def log_glob_pattern(self) -> str:
        return "rollout-*.jsonl"

    def is_session_log_path(self, path: Path, *, sessions_dir: Path | None = None) -> bool:
        return path.name.startswith("rollout-") and path.suffix == ".jsonl"

    def session_id_from_log_path(self, log_path: Path) -> str | None:
        matches = _SESSION_ID_RE.findall(log_path.name)
        return matches[-1] if matches else None

    def log_matches_session_id(self, log_path: Path, session_id: str) -> bool:
        return bool(session_id and session_id in log_path.name)

    def project_launch_defaults(self, defaults: Mapping[str, Any], *, reasoning_efforts: tuple[str, ...]) -> dict[str, Any]:
        out = dict(defaults)
        out["agent_backend"] = self.name
        out["provider_choices"] = list(out.get("model_providers") or [])
        out["reasoning_efforts"] = list(reasoning_efforts)
        out["supports_fast"] = True
        return out

    def chat_event_from_log_row(self, obj: Mapping[str, Any], *, cc_pending_tool_ids: set[str] | None = None) -> dict[str, Any] | None:
        from ..rollout_events import _codex_event_text
        from ..rollout_events import _event_ts
        from ..rollout_events import _text_message_id

        row = dict(obj)
        typ = row.get("type")
        if typ == "session_meta":
            payload = row.get("payload")
            source = payload.get("source") if isinstance(payload, dict) else None
            subagent = source.get("subagent") if isinstance(source, dict) else None
            thread_spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
            parent_thread_id = thread_spawn.get("parent_thread_id") if isinstance(thread_spawn, dict) else None
            child_thread_id = payload.get("id") if isinstance(payload, dict) else None
            if isinstance(parent_thread_id, str) and parent_thread_id and isinstance(child_thread_id, str) and child_thread_id:
                return emit_subagent_event(
                    "codex",
                    event_id=child_thread_id,
                    text=f"Subagent started (thread {child_thread_id[:8]})",
                    ts=_event_ts(row),
                )
            return None
        if typ == "event_msg":
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("invalid event_msg payload")
            payload_type = payload.get("type")
            if payload_type == "user_message":
                message = payload.get("message")
                if not isinstance(message, str):
                    return None
                ts = _event_ts(row)
                event = {"role": "user", "text": message}
                if ts is not None:
                    event["ts"] = ts
                return event
            if payload_type in ("error", "stream_error", "warning"):
                text = _codex_event_text(payload)
                if text is None:
                    return None
                ts = _event_ts(row)
                message_class = "warning" if payload_type == "warning" else "error"
                event: dict[str, Any] = {
                    "role": "assistant",
                    "text": text,
                    "message_class": message_class,
                    "message_id": _text_message_id(message_class=message_class, text=text, ts=ts),
                }
                if ts is not None:
                    event["ts"] = ts
                return event
            if payload_type == "turn_aborted":
                from ..rollout_events import _build_interrupted_event

                return _build_interrupted_event(row)
            if payload_type == "agent_message":
                # Codex emits assistant text via event_msg.agent_message (the
                # same row form idle/sidebar already treat as assistant output).
                # Project it as a visible transcript message so it renders like
                # any other assistant text and suppresses no-response injection
                # through the existing source-of-truth mechanism. Final-answer
                # phase mirrors response_item phase semantics.
                message = payload.get("message")
                if not isinstance(message, str) or not message.strip():
                    return None
                ts = _event_ts(row)
                message_class = "final_response" if payload.get("phase") == "final_answer" else "narration"
                event = {
                    "role": "assistant",
                    "text": message,
                    "message_class": message_class,
                    "message_id": _text_message_id(message_class=message_class, text=message, ts=ts),
                }
                if ts is not None:
                    event["ts"] = ts
                return event
            if payload_type in ("task_complete", "turn_complete"):
                # Codex carries the final assistant text on the turn-close row
                # via last_agent_message. Project it as a visible final_response
                # transcript message (mirroring how idle/sidebar treat it as
                # assistant output) so it renders and suppresses no-response.
                last_msg = payload.get("last_agent_message")
                if not isinstance(last_msg, str) or not last_msg.strip():
                    return None
                ts = _event_ts(row)
                message_class = "final_response"
                event = {
                    "role": "assistant",
                    "text": last_msg,
                    "message_class": message_class,
                    "message_id": _text_message_id(message_class=message_class, text=last_msg, ts=ts),
                }
                if ts is not None:
                    event["ts"] = ts
                return event
            return None

        if typ == "response_item":
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("invalid response_item payload")
            if payload.get("type") != "message" or payload.get("role") != "assistant":
                return None
            content = payload.get("content")
            if not isinstance(content, list):
                raise ValueError("invalid assistant message content")
            out_text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    out_text_parts.append(part["text"])
            if not out_text_parts:
                return None
            text = "".join(out_text_parts)
            ts = _event_ts(row)
            message_class = "final_response" if (payload.get("phase") == "final_answer" or payload.get("end_turn") is True) else "narration"
            event = {
                "role": "assistant",
                "text": text,
                "message_class": message_class,
                "message_id": _text_message_id(message_class=message_class, text=text, ts=ts),
            }
            if ts is not None:
                event["ts"] = ts
            return event

        return None

    def normalize_launch_request_options(
        self,
        obj: Mapping[str, Any],
        *,
        model: str | None,
        validation_error_type: type[ValueError],
        normalize_model_provider: Callable[..., str | None],
        normalize_preferred_auth_method: Callable[[Any], str | None],
        normalize_reasoning_effort: Callable[[Any], str | None],
        normalize_pi_reasoning_effort: Callable[..., str | None],
        normalize_cc_reasoning_effort: Callable[[Any], str | None],
        normalize_service_tier: Callable[[Any], str | None],
        codex_launch_defaults_provider: Callable[[], dict[str, Any]],
        pi_launch_defaults_provider: Callable[[], dict[str, Any]],
    ) -> dict[str, str | None]:
        allowed_providers = set(codex_launch_defaults_provider().get("model_providers") or ["openai"])
        model_provider = normalize_model_provider(
            obj.get("model_provider"),
            allowed=set(["openai", *[p for p in allowed_providers if p not in {"chatgpt", "openai-api"}]]),
        )
        return {
            "model_provider": model_provider,
            "preferred_auth_method": normalize_preferred_auth_method(obj.get("preferred_auth_method")),
            "reasoning_effort": normalize_reasoning_effort(obj.get("reasoning_effort")),
            "service_tier": normalize_service_tier(obj.get("service_tier")),
        }

    def read_run_settings_from_log(
        self,
        log_path: Path,
        *,
        read_pi_run_settings: Callable[[Path], tuple[str | None, str | None, str | None]],
        read_cc_run_settings: Callable[[Path], tuple[str | None, str | None, str | None]],
        read_session_meta_or_none_func: Callable[..., dict[str, Any] | None],
        clean_optional_text: Callable[[Any], str | None],
        display_reasoning_effort: Callable[[Any], str | None],
        find_latest_turn_context: Callable[..., Any],
    ) -> tuple[str | None, str | None, str | None]:
        meta = read_session_meta_or_none_func(log_path, agent_backend=self.name, context="run settings")
        model_provider = clean_optional_text(meta.get("model_provider")) if meta is not None else None
        model = clean_optional_text(meta.get("model")) if meta is not None else None
        reasoning_effort = display_reasoning_effort(meta.get("reasoning_effort")) if meta is not None else None
        payload = find_latest_turn_context(log_path, max_scan_bytes=8 * 1024 * 1024)
        if isinstance(payload, dict):
            # A turn context records the settings active for that turn, so its
            # model and effort are newer authority than the launch/session
            # metadata that opened the log. Missing fields retain that
            # launch-time baseline.
            context_model = clean_optional_text(payload.get("model"))
            context_effort = display_reasoning_effort(payload.get("reasoning_effort") or payload.get("effort"))
            if context_model is not None:
                model = context_model
            if context_effort is not None:
                reasoning_effort = context_effort
        return model_provider, model, reasoning_effort

    def build_launch_args(
        self,
        *,
        spawn_cwd: Path,
        codex_trust_override: str,
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
    ) -> list[str]:
        args = [
            "-c",
            codex_trust_override,
            "-c",
            "check_for_update_on_startup=false",
            "--disable",
            "goals",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if model_provider is not None:
            args.extend(["-c", f'model_provider="{model_provider}"'])
        if preferred_auth_method is not None:
            args.extend(["-c", f'preferred_auth_method="{preferred_auth_method}"'])
        if service_tier is not None:
            args.extend(["-c", f'service_tier="{service_tier}"'])
        return args

    def build_resume_args(self, *, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
        return ["resume", resume_id]

    def sessiond_working_dir(self, *, root_repo_dir: Path, requested_cwd: str) -> Path:
        return root_repo_dir

    def build_sessiond_launch_args(
        self,
        *,
        root_repo_dir: Path,
        requested_cwd: str,
        extra_args: list[str],
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
    ) -> list[str]:
        args = [
            "--no-alt-screen",
            "-c",
            "disable_response_storage=false",
            "-c",
            "disable_paste_burst=true",
            "-C",
            str(root_repo_dir),
        ]
        if requested_cwd:
            args.extend(["--add-dir", requested_cwd])
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if model_provider is not None:
            args.extend(["-c", f'model_provider="{model_provider}"'])
        if preferred_auth_method is not None:
            args.extend(["-c", f'preferred_auth_method="{preferred_auth_method}"'])
        if service_tier is not None:
            args.extend(["-c", f'service_tier="{service_tier}"'])
        args.extend(extra_args)
        return args
