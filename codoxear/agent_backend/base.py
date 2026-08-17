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

    def run_settings_from_log_rows(
        self,
        objs: list[dict[str, Any]],
        *,
        turn_context_run_settings: Callable[[Any], tuple[str | None, str | None]],
    ) -> tuple[str | None, str | None, str | None]:
        """Project the newest settings evidence in one ordered JSONL range."""
        raise NotImplementedError(f"{self.name} backend does not implement incremental run-settings extraction")

    def message_keeps_turn_busy(self, obj: Mapping[str, Any]) -> bool:
        return False

    def chat_event_from_log_row(self, obj: Mapping[str, Any], *, cc_pending_tool_ids: set[str] | None = None) -> dict[str, Any] | None:
        return None

    def project_launch_defaults(self, defaults: Mapping[str, Any], *, reasoning_efforts: tuple[str, ...]) -> dict[str, Any]:
        out = dict(defaults)
        out["agent_backend"] = self.name
        out.setdefault("provider_choices", list(out.get("provider_choices") or []))
        out["reasoning_efforts"] = list(out.get("reasoning_efforts") or reasoning_efforts)
        out.setdefault("supports_fast", False)
        return out

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
        raise NotImplementedError(f"{self.name} backend does not implement launch request normalization")

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

    def sessiond_working_dir(self, *, root_repo_dir: Path, requested_cwd: str) -> Path:
        cwd = str(requested_cwd or "").strip()
        return Path(cwd) if cwd else root_repo_dir

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
        return [
            *self.build_launch_args(
                spawn_cwd=self.sessiond_working_dir(root_repo_dir=root_repo_dir, requested_cwd=requested_cwd),
                codex_trust_override="projects.*.trust_level=\"trusted\"",
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
            ),
            *extra_args,
        ]

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
