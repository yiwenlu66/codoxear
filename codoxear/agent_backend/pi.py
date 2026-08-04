from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping

from ..subagent_events import emit_subagent_event
from .base import AgentBackend


def _compact_pi_subagent_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1].rstrip()}…"


def _pi_subagent_field(lines: list[str], label: str) -> str | None:
    prefix = f"{label.lower()}:"
    for line in lines:
        if line.lower().startswith(prefix):
            value = line[len(prefix) :].strip()
            if value:
                return value
    return None


def _pi_subagent_run_label(run_id: str | None) -> str:
    return f" (run {run_id[:8]})" if run_id else ""


def _pi_subagent_control_summary(content: str, details: Mapping[str, Any] | None) -> str | None:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None
    event = details.get("event") if isinstance(details, Mapping) else None
    event = event if isinstance(event, Mapping) else {}
    headline = lines[0].split(":", 1)[0].strip() or "Subagent status"
    agent = event.get("agent") if isinstance(event.get("agent"), str) else None
    if not agent and ":" in lines[0]:
        agent = lines[0].split(":", 1)[1].strip() or None
    run_id = event.get("runId") if isinstance(event.get("runId"), str) else _pi_subagent_field(lines, "Run")
    status = event.get("message") if isinstance(event.get("message"), str) else (_pi_subagent_field(lines, "Signal") or _pi_subagent_field(lines, "UPDATE"))
    summary = f"{headline}{f' — {agent}' if agent else ''}{_pi_subagent_run_label(run_id)}"
    if status:
        summary = f"{summary}: {status}"
    return _compact_pi_subagent_text(summary, limit=200)


def _pi_subagent_intercom_summary(content: str) -> str | None:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None
    run_id = _pi_subagent_field(lines, "Run")
    if not run_id:
        run_match = re.search(r"\brun\s+([0-9a-f-]{8,})", content, re.I)
        run_id = run_match.group(1) if run_match else None
    if "subagent needs attention" in content.lower():
        agent = None
        for line in lines:
            match = re.match(r"(.+?) needs attention in run\b", line, re.I)
            if match:
                agent = match.group(1).strip()
                break
        return _compact_pi_subagent_text(
            f"Subagent needs attention{f' — {agent}' if agent else ''}{_pi_subagent_run_label(run_id)}",
            limit=200,
        )
    status = _pi_subagent_field(lines, "Status")
    if status and status.split(maxsplit=1)[0].lower() not in {"completed", "failed", "stopped", "running"}:
        status = None
    children = _pi_subagent_field(lines, "Children")
    summary_text = None
    for index, line in enumerate(lines):
        if line.rstrip(":").lower() == "summary":
            summary_text = next((candidate for candidate in lines[index + 1 :] if candidate), None)
            break
    summary = f"Subagent result{f' — {status}' if status else ''}{_pi_subagent_run_label(run_id)}"
    if children:
        summary = f"{summary}; {children}"
    if summary_text:
        summary = f"{summary}: {summary_text}"
    return _compact_pi_subagent_text(summary, limit=200)


class PiBackend(AgentBackend):
    def is_session_log_path(self, path: Path, *, sessions_dir: Path | None = None) -> bool:
        if path.suffix != ".jsonl":
            return False
        if sessions_dir is None:
            return "/.pi/agent/sessions/" in str(path).replace("\\", "/")
        try:
            path.resolve().relative_to(sessions_dir.resolve())
        except Exception:
            return False
        return True

    def session_id_from_log_path(self, log_path: Path) -> str | None:
        from ..pi_log import read_pi_session_id

        return read_pi_session_id(log_path)

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
        return read_pi_run_settings(log_path)

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
        pi_launch_defaults = pi_launch_defaults_provider()
        model_provider = normalize_model_provider(obj.get("model_provider"), allowed=None)
        if obj.get("preferred_auth_method") not in (None, ""):
            raise validation_error_type(f"preferred_auth_method is not supported for {self.name}")
        if obj.get("service_tier") not in (None, ""):
            raise validation_error_type("service_tier is not supported for pi")
        return {
            "model_provider": model_provider,
            "preferred_auth_method": None,
            "reasoning_effort": normalize_pi_reasoning_effort(
                obj.get("reasoning_effort"),
                model_provider=model_provider,
                model=model,
                reasoning_efforts_by_model=pi_launch_defaults.get("reasoning_efforts_by_model") if isinstance(pi_launch_defaults, dict) else None,
            ),
            "service_tier": None,
        }

    def message_keeps_turn_busy(self, obj: Mapping[str, Any]) -> bool:
        from ..pi_log import pi_assistant_is_terminal_no_visible_response
        from ..pi_log import pi_assistant_thinking_count
        from ..pi_log import pi_assistant_tool_use_count
        from ..pi_log import pi_message_role

        row = dict(obj)
        if pi_assistant_is_terminal_no_visible_response(row):
            return False
        role = pi_message_role(row)
        if role == "toolResult":
            return True
        return (pi_assistant_thinking_count(row) > 0) or (pi_assistant_tool_use_count(row) > 0)

    def chat_event_from_log_row(self, obj: Mapping[str, Any], *, cc_pending_tool_ids: set[str] | None = None) -> dict[str, Any] | None:
        from ..pi_log import pi_assistant_error_text
        from ..pi_log import pi_assistant_is_aborted_turn
        from ..pi_log import pi_assistant_is_final_turn_end
        from ..pi_log import pi_assistant_is_terminal_no_visible_response
        from ..pi_log import pi_assistant_text
        from ..pi_log import pi_user_is_agent_internal_delivery
        from ..pi_log import pi_user_text
        from ..rollout_events import _event_ts
        from ..rollout_events import _text_message_id

        row = dict(obj)
        row_type = row.get("type")
        # Pi writes harness coordination rows for subagent progress and
        # results.  Normalize their user-meaningful summaries as assistant
        # narration; generic coordination traffic remains private.
        if row_type == "active_long_running":
            return None
        if row_type == "custom_message":
            custom_type = row.get("customType")
            content = row.get("content")
            if not isinstance(content, str):
                return None
            if custom_type == "subagent_control_notice":
                summary = _pi_subagent_control_summary(
                    content,
                    row.get("details") if isinstance(row.get("details"), Mapping) else None,
                )
            elif custom_type == "intercom_message":
                summary = _pi_subagent_intercom_summary(content)
            else:
                summary = None
            return self._subagent_narration_event(row, summary) if summary else None
        if row_type != "message":
            return None
        user_text = pi_user_text(row)
        if isinstance(user_text, str) and user_text:
            ts = _event_ts(row)
            event: dict[str, Any] = {"role": "user", "text": user_text}
            if pi_user_is_agent_internal_delivery(row):
                event["agent_internal_delivery"] = True
            if ts is not None:
                event["ts"] = ts
            return event
        if pi_assistant_is_aborted_turn(row):
            from ..rollout_events import _build_interrupted_event

            return _build_interrupted_event(row, partial_text=pi_assistant_text(row))
        assistant_text = pi_assistant_text(row)
        if isinstance(assistant_text, str) and assistant_text:
            ts = _event_ts(row)
            message_class = "final_response" if pi_assistant_is_final_turn_end(row) else "narration"
            event = {
                "role": "assistant",
                "text": assistant_text,
                "message_class": message_class,
                "message_id": _text_message_id(message_class=message_class, text=assistant_text, ts=ts),
            }
            if ts is not None:
                event["ts"] = ts
            return event
        error_text = pi_assistant_error_text(row)
        if isinstance(error_text, str) and error_text:
            ts = _event_ts(row)
            return {
                "role": "assistant",
                "text": error_text,
                "message_class": "error",
                "message_id": _text_message_id(message_class="error", text=error_text, ts=ts),
                **({"ts": ts} if ts is not None else {}),
            }
        if pi_assistant_is_terminal_no_visible_response(row):
            from ..rollout_chat_events import _build_no_response_event

            return _build_no_response_event(row)
        return None

    @staticmethod
    def _subagent_narration_event(row: Mapping[str, Any], text: str) -> dict[str, Any]:
        from ..rollout_events import _event_ts
        from ..rollout_events import _text_message_id

        ts = _event_ts(dict(row))
        row_id = row.get("id")
        event_id = (
            row_id
            if isinstance(row_id, str) and row_id
            else _text_message_id(message_class="narration", text=text, ts=ts)
        )
        return emit_subagent_event("pi", event_id=event_id, text=text, ts=ts)

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
        if preferred_auth_method is not None:
            raise ValueError("preferred_auth_method is not supported for pi")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for pi")
        args: list[str] = []
        if model_provider is not None:
            args.extend(["--provider", model_provider])
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["--thinking", reasoning_effort])
        return args

    def build_resume_args(self, *, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
        resume_target = str((resume_row or {}).get("log_path") or "").strip()
        return ["--session", resume_target or resume_id]
