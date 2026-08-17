from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from ..app_dir_runtime import resolve_default_app_dir
from .base import AgentBackend


class ClaudeCodeBackend(AgentBackend):
    def apply_launch_environment(
        self,
        env: dict[str, str],
        *,
        homes: Mapping[str, str | Path],
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        resume_session_id: str | None = None,
    ) -> dict[str, str]:
        out = super().apply_launch_environment(
            env,
            homes=homes,
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            resume_session_id=resume_session_id,
        )
        out["CODEX_WEB_CC_SUBAGENT_RUNS_ROOT"] = str(resolve_default_app_dir(legacy_warned=False).app_dir / "cc-subagent-runs")
        return out

    @staticmethod
    def _subagent_hook_settings() -> str:
        command = shlex.join([sys.executable, "-m", "codoxear.cc_subagents"])
        return json.dumps(
            {
                "hooks": {
                    event_name: [
                        {
                            "matcher": "*",
                            "hooks": [{"type": "command", "command": command, "timeout": 5}],
                        }
                    ]
                    for event_name in ("SubagentStart", "SubagentStop")
                }
            },
            separators=(",", ":"),
        )

    def is_session_log_path(self, path: Path, *, sessions_dir: Path | None = None) -> bool:
        if path.suffix != ".jsonl":
            return False
        path_text = str(path).replace("\\", "/")
        if "/subagents/" in path_text:
            return False
        if path.name == "history.jsonl":
            return False
        if sessions_dir is None:
            return "/.claude/projects/" in path_text
        try:
            path.resolve().relative_to(sessions_dir.resolve())
        except Exception:
            return False
        return True

    def session_id_from_log_path(self, log_path: Path) -> str | None:
        from ..cc_log import read_cc_session_id

        return read_cc_session_id(log_path)

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
        return read_cc_run_settings(log_path)

    def run_settings_from_log_rows(
        self,
        objs: list[dict[str, Any]],
        *,
        turn_context_run_settings: Callable[[Any], tuple[str | None, str | None]],
    ) -> tuple[str | None, str | None, str | None]:
        del turn_context_run_settings
        for obj in reversed(objs):
            if not isinstance(obj, dict) or obj.get("type") != "assistant":
                continue
            message = obj.get("message")
            model = message.get("model") if isinstance(message, dict) else None
            if isinstance(model, str) and model.strip():
                return None, model.strip(), None
        return None, None, None

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
        if obj.get("model_provider") not in (None, ""):
            raise validation_error_type("model_provider is not supported for cc")
        if obj.get("preferred_auth_method") not in (None, ""):
            raise validation_error_type(f"preferred_auth_method is not supported for {self.name}")
        if obj.get("service_tier") not in (None, ""):
            raise validation_error_type("service_tier is not supported for cc")
        return {
            "model_provider": None,
            "preferred_auth_method": None,
            "reasoning_effort": normalize_cc_reasoning_effort(obj.get("reasoning_effort")),
            "service_tier": None,
        }

    def message_keeps_turn_busy(self, obj: Mapping[str, Any]) -> bool:
        from ..cc_log import cc_assistant_thinking_count
        from ..cc_log import cc_assistant_tool_use_count
        from ..cc_log import cc_message_role

        row = dict(obj)
        role = cc_message_role(row)
        if role == "toolResult":
            return True
        return (cc_assistant_thinking_count(row) > 0) or (cc_assistant_tool_use_count(row) > 0)

    def chat_event_from_log_row(self, obj: Mapping[str, Any], *, cc_pending_tool_ids: set[str] | None = None) -> dict[str, Any] | None:
        from ..cc_log import cc_apply_tool_result_to_pending
        from ..cc_log import cc_assistant_is_api_error
        from ..cc_log import cc_assistant_is_final_turn_end
        from ..cc_log import cc_assistant_pending_tool_use_ids
        from ..cc_log import cc_assistant_text
        from ..cc_log import cc_assistant_tool_use_count
        from ..cc_log import cc_message_role
        from ..cc_log import cc_system_api_error_is_terminal
        from ..cc_log import cc_system_api_error_text
        from ..cc_log import cc_user_text
        from ..rollout_events import _event_ts
        from ..rollout_events import _text_message_id

        row = dict(obj)
        typ = row.get("type")
        if typ == "user":
            user_text = cc_user_text(row)
            if isinstance(user_text, str) and user_text:
                if cc_pending_tool_ids is not None:
                    cc_pending_tool_ids.clear()
                ts = _event_ts(row)
                event: dict[str, Any] = {"role": "user", "text": user_text}
                if ts is not None:
                    event["ts"] = ts
                return event
            if cc_pending_tool_ids is not None and cc_message_role(row) == "toolResult":
                cc_apply_tool_result_to_pending(row, cc_pending_tool_ids)
            return None

        if typ == "assistant":
            if cc_pending_tool_ids is not None and cc_assistant_tool_use_count(row) > 0:
                cc_pending_tool_ids.update(cc_assistant_pending_tool_use_ids(row))
            assistant_text = cc_assistant_text(row)
            if isinstance(assistant_text, str) and assistant_text:
                ts = _event_ts(row)
                if cc_assistant_is_api_error(row):
                    message_class = "error"
                else:
                    message_class = "final_response" if cc_assistant_is_final_turn_end(row) and not cc_pending_tool_ids else "narration"
                event = {
                    "role": "assistant",
                    "text": assistant_text,
                    "message_class": message_class,
                    "message_id": _text_message_id(message_class=message_class, text=assistant_text, ts=ts),
                }
                if ts is not None:
                    event["ts"] = ts
                return event
            return None

        if typ == "system" and cc_system_api_error_is_terminal(row):
            error_text = cc_system_api_error_text(row)
            if isinstance(error_text, str) and error_text:
                ts = _event_ts(row)
                event = {
                    "role": "assistant",
                    "text": error_text,
                    "message_class": "error",
                    "message_id": _text_message_id(message_class="error", text=error_text, ts=ts),
                }
                if ts is not None:
                    event["ts"] = ts
                return event
            return None

        return None

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
        if model_provider is not None:
            raise ValueError("model_provider is not supported for cc")
        if preferred_auth_method is not None:
            raise ValueError("preferred_auth_method is not supported for cc")
        if service_tier is not None:
            raise ValueError("service_tier is not supported for cc")
        args = ["--dangerously-skip-permissions", "--settings", self._subagent_hook_settings()]
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["--effort", reasoning_effort])
        return args

    def build_resume_args(self, *, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
        return ["--resume", resume_id]
