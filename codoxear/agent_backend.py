from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


_REQUEST_ENV_VARS = (
    "CODEX_WEB_MODEL_PROVIDER",
    "CODEX_WEB_PREFERRED_AUTH_METHOD",
    "CODEX_WEB_MODEL",
    "CODEX_WEB_REASONING_EFFORT",
    "CODEX_WEB_SERVICE_TIER",
    "CODEX_WEB_TRANSPORT",
    "CODEX_WEB_TMUX_SESSION",
    "CODEX_WEB_TMUX_WINDOW",
    "CODEX_WEB_LAUNCH_ID",
    "CODEX_WEB_SPAWN_NONCE",
    "CODEX_WEB_RESUME_SESSION_ID",
    "CODEX_WEB_RESUME_LOG_PATH",
)

_BACKEND_HOME_ENV_VARS = ("CODEX_HOME", "PI_HOME", "CLAUDE_CONFIG_DIR")
_BACKEND_BIN_ENV_VARS = ("CODEX_BIN", "PI_BIN", "CLAUDE_BIN")
_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


@dataclass(frozen=True)
class AgentBackend:
    name: str
    bin_env_var: str
    home_env_var: str
    default_bin: str
    default_home_dirname: str
    sessions_relpath: tuple[str, ...]

    def cli_bin(self, env: dict[str, str] | None = None) -> str:
        env_map = os.environ if env is None else env
        value = str(env_map.get(self.bin_env_var) or "").strip()
        return value or self.default_bin

    def home(self, env: dict[str, str] | None = None) -> Path:
        env_map = os.environ if env is None else env
        raw = str(env_map.get(self.home_env_var) or "").strip()
        if raw:
            return Path(raw).expanduser()
        return Path.home() / self.default_home_dirname

    def sessions_dir(self, env: dict[str, str] | None = None) -> Path:
        return self.home(env).joinpath(*self.sessions_relpath)

    def log_glob_pattern(self) -> str:
        return "*.jsonl"

    def is_session_log_path(self, path: Path, *, sessions_dir: Path | None = None) -> bool:
        raise NotImplementedError(f"{self.name} backend does not implement log path recognition")

    def session_id_from_log_path(self, log_path: Path) -> str | None:
        return None

    def log_matches_session_id(self, log_path: Path, session_id: str) -> bool:
        return self.session_id_from_log_path(log_path) == session_id

    def session_id_from_payload_or_log(self, log_path: Path, payload: Mapping[str, Any]) -> str | None:
        raw = payload.get("id")
        return raw if isinstance(raw, str) and raw else self.session_id_from_log_path(log_path)

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
        raise NotImplementedError(f"{self.name} backend does not implement run-settings extraction")

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
        raise NotImplementedError(f"{self.name} backend does not implement launch args")

    def build_resume_args(self, *, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
        raise NotImplementedError(f"{self.name} backend does not implement resume args")

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
        if self.name not in homes:
            raise ValueError(f"missing home path for {self.name}")

        env["CODEX_WEB_OWNER"] = "web"
        env["CODEX_WEB_AGENT_BACKEND"] = self.name
        env.setdefault(self.home_env_var, str(homes[self.name]))
        for key in _BACKEND_HOME_ENV_VARS:
            if key != self.home_env_var:
                env.pop(key, None)
        for key in _REQUEST_ENV_VARS:
            env.pop(key, None)

        self._apply_request_environment(
            env,
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            resume_session_id=resume_session_id,
        )
        return env

    def build_tmux_inline_env(
        self,
        env: Mapping[str, str],
        *,
        tmux_session: str,
        tmux_window: str,
        launch_id: str,
        spawn_nonce: str,
        resume_session_id: str | None = None,
        model_provider: str | None = None,
        preferred_auth_method: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        inherited_backend_bin: str | None = None,
    ) -> dict[str, str]:
        inline_env = {
            "CODEX_WEB_OWNER": "web",
            "CODEX_WEB_AGENT_BACKEND": self.name,
            "CODEX_WEB_TRANSPORT": "tmux",
            "CODEX_WEB_TMUX_SESSION": tmux_session,
            "CODEX_WEB_TMUX_WINDOW": tmux_window,
            "CODEX_WEB_LAUNCH_ID": launch_id,
            "CODEX_WEB_SPAWN_NONCE": spawn_nonce,
            self.home_env_var: str(env[self.home_env_var]),
        }
        self._apply_request_environment(
            inline_env,
            model_provider=model_provider,
            preferred_auth_method=preferred_auth_method,
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            resume_session_id=resume_session_id,
        )
        if inherited_backend_bin is not None:
            inline_env[self.bin_env_var] = inherited_backend_bin
        return inline_env

    @staticmethod
    def tmux_unset_vars() -> list[str]:
        return [
            *_BACKEND_HOME_ENV_VARS,
            *_BACKEND_BIN_ENV_VARS,
            "CODEX_WEB_OWNER",
            "CODEX_WEB_AGENT_BACKEND",
            *_REQUEST_ENV_VARS,
        ]

    @staticmethod
    def _apply_request_environment(
        env: dict[str, str],
        *,
        model_provider: str | None,
        preferred_auth_method: str | None,
        model: str | None,
        reasoning_effort: str | None,
        service_tier: str | None,
        resume_session_id: str | None,
    ) -> None:
        if model_provider is not None:
            env["CODEX_WEB_MODEL_PROVIDER"] = model_provider
        if preferred_auth_method is not None:
            env["CODEX_WEB_PREFERRED_AUTH_METHOD"] = preferred_auth_method
        if model is not None:
            env["CODEX_WEB_MODEL"] = model
        if reasoning_effort is not None:
            env["CODEX_WEB_REASONING_EFFORT"] = reasoning_effort
        if service_tier is not None:
            env["CODEX_WEB_SERVICE_TIER"] = service_tier
        if resume_session_id is not None:
            env["CODEX_WEB_RESUME_SESSION_ID"] = resume_session_id


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
        if model is None or reasoning_effort is None:
            payload = find_latest_turn_context(log_path, max_scan_bytes=8 * 1024 * 1024)
            if isinstance(payload, dict):
                if model is None:
                    model = clean_optional_text(payload.get("model"))
                if reasoning_effort is None:
                    reasoning_effort = display_reasoning_effort(payload.get("reasoning_effort") or payload.get("effort"))
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
        from .pi_log import read_pi_session_id

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


class ClaudeCodeBackend(AgentBackend):
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
        from .cc_log import read_cc_session_id

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
        args = ["--dangerously-skip-permissions"]
        if model is not None:
            args.extend(["--model", model])
        if reasoning_effort is not None:
            args.extend(["--effort", reasoning_effort])
        return args

    def build_resume_args(self, *, resume_id: str, resume_row: Mapping[str, Any] | None = None) -> list[str]:
        return ["--resume", resume_id]


CODEX_BACKEND = CodexBackend(
    name="codex",
    bin_env_var="CODEX_BIN",
    home_env_var="CODEX_HOME",
    default_bin="codex",
    default_home_dirname=".codex",
    sessions_relpath=("sessions",),
)

PI_BACKEND = PiBackend(
    name="pi",
    bin_env_var="PI_BIN",
    home_env_var="PI_HOME",
    default_bin="pi",
    default_home_dirname=".pi",
    sessions_relpath=("agent", "sessions"),
)

CC_BACKEND = ClaudeCodeBackend(
    name="cc",
    bin_env_var="CLAUDE_BIN",
    home_env_var="CLAUDE_CONFIG_DIR",
    default_bin="claude",
    default_home_dirname=".claude",
    sessions_relpath=("projects",),
)

_BACKENDS: dict[str, AgentBackend] = {
    CODEX_BACKEND.name: CODEX_BACKEND,
    PI_BACKEND.name: PI_BACKEND,
    CC_BACKEND.name: CC_BACKEND,
}


def normalize_agent_backend(value: object, *, default: str = "codex") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = default
    if raw not in _BACKENDS:
        allowed = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"agent_backend must be one of {allowed}")
    return raw


def get_agent_backend(value: object, *, default: str = "codex") -> AgentBackend:
    return _BACKENDS[normalize_agent_backend(value, default=default)]


def infer_agent_backend_from_log_path(path: Path) -> str | None:
    if CODEX_BACKEND.is_session_log_path(path):
        return "codex"
    if PI_BACKEND.is_session_log_path(path):
        return "pi"
    if CC_BACKEND.is_session_log_path(path):
        return "cc"
    if CC_BACKEND.is_session_log_path(path, sessions_dir=CC_BACKEND.sessions_dir()):
        return "cc"
    return None
