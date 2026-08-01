from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_backend import get_agent_backend, normalize_agent_backend
from .launch_path_runtime import load_env_file
from .server_routing import normalize_url_prefix
from .util import default_app_dir, launch_attempts_path


SERVER_CONFIG_EXPORT_NAMES = (
    "APP_DIR",
    "PROC_ROOT",
    "SOCK_DIR",
    "STATE_PATH",
    "HMAC_SECRET_PATH",
    "LAUNCH_ATTEMPTS_PATH",
    "UPLOAD_DIR",
    "UNATTENDED_PATH",
    "ALIAS_PATH",
    "SIDEBAR_META_PATH",
    "HIDDEN_SESSIONS_PATH",
    "FILE_HISTORY_PATH",
    "VIDEO_PREVIEW_DIR",
    "QUEUE_PATH",
    "PENDING_ATTACHMENTS_PATH",
    "STAGED_ATTACHMENTS_PATH",
    "COMMIT_UNKNOWN_SENDS_PATH",
    "RECENT_CWD_PATH",
    "VOICE_SETTINGS_PATH",
    "UNATTENDED_PROMPT_PATH",
    "PUSH_SUBSCRIPTIONS_PATH",
    "DELIVERY_LEDGER_PATH",
    "VAPID_PRIVATE_KEY_PATH",
    "COOKIE_NAME",
    "COOKIE_EXPIRES",
    "COOKIE_SECURE",
    "URL_PREFIX",
    "COOKIE_PATH",
    "TMUX_SESSION_NAME",
    "TMUX_META_WAIT_SECONDS",
    "TMUX_AVAILABLE_TTL_SECONDS",
    "CODEX_HOME",
    "CODEX_SESSIONS_DIR",
    "CODEX_CONFIG_PATH",
    "MODELS_CACHE_PATH",
    "PI_HOME",
    "PI_SESSIONS_DIR",
    "PI_SETTINGS_PATH",
    "PI_MODELS_PATH",
    "PI_AUTH_PATH",
    "CC_HOME",
    "CC_SESSIONS_DIR",
    "CC_SETTINGS_PATH",
    "DEFAULT_AGENT_BACKEND",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "UNATTENDED_DEFAULT_IDLE_MINUTES",
    "UNATTENDED_DEFAULT_MAX_INJECTIONS",
    "UNATTENDED_SWEEP_SECONDS",
    "QUEUE_SWEEP_SECONDS",
    "QUEUE_SWEEP_MAX_DRAINS",
    "QUEUE_SWEEP_MAX_ATTEMPTS",
    "VOICE_PUSH_SWEEP_SECONDS",
    "QUEUE_IDLE_GRACE_SECONDS",
    "UNATTENDED_MAX_SCAN_BYTES",
    "DISCOVER_MIN_INTERVAL_SECONDS",
    "METRICS_WINDOW",
    "FILE_HISTORY_MAX",
    "GIT_DIFF_MAX_BYTES",
    "GIT_DIFF_TIMEOUT_SECONDS",
    "GIT_WORKTREE_TIMEOUT_SECONDS",
    "GIT_CHANGED_FILES_MAX",
    "ATTACH_UPLOAD_MAX_BYTES",
    "ATTACH_UPLOAD_BODY_MAX_BYTES",
    "SEND_COMMIT_TIMEOUT_SECONDS",
    "COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS",
    "SIDEBAR_PRIORITY_HALF_LIFE_SECONDS",
    "SIDEBAR_PRIORITY_BUCKET_SECONDS",
    "RECENT_CWD_MAX",
    "TRANSCRIPT_EXPORT_MAX_BYTES",
)


@dataclass(frozen=True)
class ServerConfig:
    APP_DIR: Path
    PROC_ROOT: Path
    SOCK_DIR: Path
    STATE_PATH: Path
    HMAC_SECRET_PATH: Path
    LAUNCH_ATTEMPTS_PATH: Path
    UPLOAD_DIR: Path
    UNATTENDED_PATH: Path
    ALIAS_PATH: Path
    SIDEBAR_META_PATH: Path
    HIDDEN_SESSIONS_PATH: Path
    FILE_HISTORY_PATH: Path
    VIDEO_PREVIEW_DIR: Path
    QUEUE_PATH: Path
    PENDING_ATTACHMENTS_PATH: Path
    STAGED_ATTACHMENTS_PATH: Path
    COMMIT_UNKNOWN_SENDS_PATH: Path
    RECENT_CWD_PATH: Path
    VOICE_SETTINGS_PATH: Path
    UNATTENDED_PROMPT_PATH: Path
    PUSH_SUBSCRIPTIONS_PATH: Path
    DELIVERY_LEDGER_PATH: Path
    VAPID_PRIVATE_KEY_PATH: Path
    DOTENV_PATH: Path
    COOKIE_NAME: str
    COOKIE_EXPIRES: str
    COOKIE_SECURE: bool
    URL_PREFIX: str
    COOKIE_PATH: str
    TMUX_SESSION_NAME: str
    TMUX_META_WAIT_SECONDS: float
    TMUX_AVAILABLE_TTL_SECONDS: float
    CODEX_HOME: Path
    CODEX_SESSIONS_DIR: Path
    CODEX_CONFIG_PATH: Path
    MODELS_CACHE_PATH: Path
    PI_HOME: Path
    PI_SESSIONS_DIR: Path
    PI_SETTINGS_PATH: Path
    PI_MODELS_PATH: Path
    PI_AUTH_PATH: Path
    CC_HOME: Path
    CC_SESSIONS_DIR: Path
    CC_SETTINGS_PATH: Path
    DEFAULT_AGENT_BACKEND: str
    DEFAULT_HOST: str
    DEFAULT_PORT: int
    UNATTENDED_DEFAULT_IDLE_MINUTES: int
    UNATTENDED_DEFAULT_MAX_INJECTIONS: int
    UNATTENDED_SWEEP_SECONDS: float
    QUEUE_SWEEP_SECONDS: float
    QUEUE_SWEEP_MAX_DRAINS: int
    QUEUE_SWEEP_MAX_ATTEMPTS: int
    VOICE_PUSH_SWEEP_SECONDS: float
    QUEUE_IDLE_GRACE_SECONDS: float
    UNATTENDED_MAX_SCAN_BYTES: int
    DISCOVER_MIN_INTERVAL_SECONDS: float
    METRICS_WINDOW: int
    FILE_HISTORY_MAX: int
    GIT_DIFF_MAX_BYTES: int
    GIT_DIFF_TIMEOUT_SECONDS: float
    GIT_WORKTREE_TIMEOUT_SECONDS: float
    GIT_CHANGED_FILES_MAX: int
    ATTACH_UPLOAD_MAX_BYTES: int
    ATTACH_UPLOAD_BODY_MAX_BYTES: int
    SEND_COMMIT_TIMEOUT_SECONDS: float
    COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS: float
    SIDEBAR_PRIORITY_HALF_LIFE_SECONDS: float
    SIDEBAR_PRIORITY_BUCKET_SECONDS: float
    RECENT_CWD_MAX: int
    TRANSCRIPT_EXPORT_MAX_BYTES: int


def export_server_config(target: MutableMapping[str, Any], config: ServerConfig) -> None:
    target["_DOTENV"] = config.DOTENV_PATH
    for name in SERVER_CONFIG_EXPORT_NAMES:
        target[name] = getattr(config, name)



def apply_dotenv(
    dotenv_path: Path,
    *,
    environ: MutableMapping[str, str],
    load_env_file_func: Any = load_env_file,
) -> None:
    if not dotenv_path.exists():
        return
    for key, value in load_env_file_func(dotenv_path).items():
        environ.setdefault(key, value)


def _env_float(environ: MutableMapping[str, str], name: str, default: str) -> float:
    return float(environ.get(name, default))


def _env_int(environ: MutableMapping[str, str], name: str, default: str) -> int:
    return int(environ.get(name, default))


def _env_bool_1(environ: MutableMapping[str, str], name: str, default: str = "0") -> bool:
    return environ.get(name, default) == "1"

# RFC 6265 §4.1.1: cookie-name = token
# token = 1*<any CHAR except CTLs or separators>
# CTLs = %x00-1F / %x7F
# separators = ( ) < > @ , ; : \ " / [ ] ? = { } SP HT
# SP is 0x20; HT (0x09) is already covered by CTL.
_COOKIE_NAME_SEPARATORS = frozenset('()<>@,;:\\"/[]?={} ')


def _validate_cookie_name(name: str) -> str:
    """Return *name* if it is a valid RFC 6265 cookie-name, otherwise raise ValueError."""
    if not name:
        raise ValueError("CODEX_WEB_COOKIE_NAME must not be empty")
    for offset, ch in enumerate(name):
        cp = ord(ch)
        if cp <= 0x1F or cp == 0x7F or ch in _COOKIE_NAME_SEPARATORS:
            raise ValueError(f"CODEX_WEB_COOKIE_NAME contains invalid character U+{cp:04X} at offset {offset}")
    return name


def build_server_config(
    *,
    cwd: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> ServerConfig:
    env = os.environ if environ is None else environ
    app_dir = default_app_dir()
    dotenv_path = ((Path.cwd() if cwd is None else cwd) / ".env").resolve()
    apply_dotenv(dotenv_path, environ=env)

    cookie_name_raw = env.get("CODEX_WEB_COOKIE_NAME", "")
    cookie_name = "codoxear_auth" if cookie_name_raw == "" else _validate_cookie_name(cookie_name_raw)
    url_prefix = normalize_url_prefix(env.get("CODEX_WEB_URL_PREFIX"))
    cookie_path = (url_prefix + "/") if url_prefix else "/"
    tmux_session_name = (env.get("CODEX_WEB_TMUX_SESSION") or "codoxear").strip() or "codoxear"

    codex_home_env = env.get("CODEX_HOME")
    if codex_home_env is None or (not codex_home_env.strip()):
        codex_home = Path.home() / ".codex"
    else:
        codex_home = Path(codex_home_env)
    pi_home = get_agent_backend("pi").home(env)
    cc_home = get_agent_backend("cc").home(env)

    attach_upload_max_bytes = _env_int(env, "CODEX_WEB_ATTACH_MAX_BYTES", str(16 * 1024 * 1024))
    attach_upload_body_max_bytes = _env_int(
        env,
        "CODEX_WEB_ATTACH_BODY_MAX_BYTES",
        str((4 * ((attach_upload_max_bytes + 2) // 3)) + (64 * 1024)),
    )
    queue_sweep_max_drains = max(1, _env_int(env, "CODEX_WEB_QUEUE_SWEEP_MAX_DRAINS", "4"))
    queue_sweep_max_attempts = max(queue_sweep_max_drains, _env_int(env, "CODEX_WEB_QUEUE_SWEEP_MAX_ATTEMPTS", "16"))

    return ServerConfig(
        APP_DIR=app_dir,
        PROC_ROOT=Path("/proc"),
        SOCK_DIR=app_dir / "socks",
        STATE_PATH=app_dir / "state.json",
        HMAC_SECRET_PATH=app_dir / "hmac_secret",
        LAUNCH_ATTEMPTS_PATH=launch_attempts_path(app_dir),
        UPLOAD_DIR=app_dir / "uploads",
        UNATTENDED_PATH=app_dir / "unattended.json",
        ALIAS_PATH=app_dir / "session_aliases.json",
        SIDEBAR_META_PATH=app_dir / "session_sidebar.json",
        HIDDEN_SESSIONS_PATH=app_dir / "hidden_sessions.json",
        FILE_HISTORY_PATH=app_dir / "session_files.json",
        VIDEO_PREVIEW_DIR=app_dir / "video_previews",
        QUEUE_PATH=app_dir / "session_queues.json",
        PENDING_ATTACHMENTS_PATH=app_dir / "pending_attachments.json",
        STAGED_ATTACHMENTS_PATH=app_dir / "staged_attachments.json",
        COMMIT_UNKNOWN_SENDS_PATH=app_dir / "commit_unknown_sends.json",
        RECENT_CWD_PATH=app_dir / "recent_cwds.json",
        VOICE_SETTINGS_PATH=app_dir / "voice_settings.json",
        UNATTENDED_PROMPT_PATH=app_dir / "unattended_prompt.txt",
        PUSH_SUBSCRIPTIONS_PATH=app_dir / "push_subscriptions.json",
        DELIVERY_LEDGER_PATH=app_dir / "voice_delivery_ledger.json",
        VAPID_PRIVATE_KEY_PATH=app_dir / "webpush_vapid_private.pem",
        DOTENV_PATH=dotenv_path,
        COOKIE_NAME=cookie_name,
        COOKIE_EXPIRES="Fri, 31 Dec 9999 23:59:59 GMT",
        COOKIE_SECURE=_env_bool_1(env, "CODEX_WEB_COOKIE_SECURE"),
        URL_PREFIX=url_prefix,
        COOKIE_PATH=cookie_path,
        TMUX_SESSION_NAME=tmux_session_name,
        TMUX_META_WAIT_SECONDS=3.0,
        TMUX_AVAILABLE_TTL_SECONDS=_env_float(env, "CODEX_WEB_TMUX_AVAILABLE_TTL_SECONDS", "30.0"),
        CODEX_HOME=codex_home,
        CODEX_SESSIONS_DIR=codex_home / "sessions",
        CODEX_CONFIG_PATH=codex_home / "config.toml",
        MODELS_CACHE_PATH=codex_home / "models_cache.json",
        PI_HOME=pi_home,
        PI_SESSIONS_DIR=get_agent_backend("pi").sessions_dir(env),
        PI_SETTINGS_PATH=pi_home / "agent" / "settings.json",
        PI_MODELS_PATH=pi_home / "agent" / "models.json",
        PI_AUTH_PATH=pi_home / "agent" / "auth.json",
        CC_HOME=cc_home,
        CC_SESSIONS_DIR=get_agent_backend("cc").sessions_dir(env),
        CC_SETTINGS_PATH=cc_home / "settings.json",
        DEFAULT_AGENT_BACKEND=normalize_agent_backend(env.get("CODEX_WEB_DEFAULT_AGENT_BACKEND"), default="pi"),
        DEFAULT_HOST=env.get("CODEX_WEB_HOST", "::"),
        DEFAULT_PORT=_env_int(env, "CODEX_WEB_PORT", "8743"),
        UNATTENDED_DEFAULT_IDLE_MINUTES=5,
        UNATTENDED_DEFAULT_MAX_INJECTIONS=10,
        UNATTENDED_SWEEP_SECONDS=_env_float(env, "CODEX_WEB_UNATTENDED_SWEEP_SECONDS", "2.5"),
        QUEUE_SWEEP_SECONDS=_env_float(env, "CODEX_WEB_QUEUE_SWEEP_SECONDS", "1.0"),
        QUEUE_SWEEP_MAX_DRAINS=queue_sweep_max_drains,
        QUEUE_SWEEP_MAX_ATTEMPTS=queue_sweep_max_attempts,
        VOICE_PUSH_SWEEP_SECONDS=_env_float(env, "CODEX_WEB_VOICE_PUSH_SWEEP_SECONDS", "1.0"),
        QUEUE_IDLE_GRACE_SECONDS=_env_float(env, "CODEX_WEB_QUEUE_IDLE_GRACE_SECONDS", "10.0"),
        UNATTENDED_MAX_SCAN_BYTES=_env_int(env, "CODEX_WEB_UNATTENDED_MAX_SCAN_BYTES", str(8 * 1024 * 1024)),
        DISCOVER_MIN_INTERVAL_SECONDS=_env_float(env, "CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS", "1.0"),
        METRICS_WINDOW=_env_int(env, "CODEX_WEB_METRICS_WINDOW", "256"),
        FILE_HISTORY_MAX=_env_int(env, "CODEX_WEB_FILE_HISTORY_MAX", "20"),
        GIT_DIFF_MAX_BYTES=_env_int(env, "CODEX_WEB_GIT_DIFF_MAX_BYTES", str(800 * 1024)),
        GIT_DIFF_TIMEOUT_SECONDS=_env_float(env, "CODEX_WEB_GIT_DIFF_TIMEOUT_SECONDS", "4.0"),
        GIT_WORKTREE_TIMEOUT_SECONDS=_env_float(env, "CODEX_WEB_GIT_WORKTREE_TIMEOUT_SECONDS", "10.0"),
        GIT_CHANGED_FILES_MAX=_env_int(env, "CODEX_WEB_GIT_CHANGED_FILES_MAX", "400"),
        ATTACH_UPLOAD_MAX_BYTES=attach_upload_max_bytes,
        ATTACH_UPLOAD_BODY_MAX_BYTES=attach_upload_body_max_bytes,
        SEND_COMMIT_TIMEOUT_SECONDS=_env_float(env, "CODEX_WEB_SEND_COMMIT_TIMEOUT_SECONDS", "30"),
        COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS=_env_float(env, "CODEX_WEB_COMMIT_UNKNOWN_ORPHAN_PRUNE_SECONDS", str(7 * 24 * 3600)),
        SIDEBAR_PRIORITY_HALF_LIFE_SECONDS=8.0 * 3600.0,
        SIDEBAR_PRIORITY_BUCKET_SECONDS=_env_float(env, "CODEX_WEB_SIDEBAR_PRIORITY_BUCKET_SECONDS", "30.0"),
        RECENT_CWD_MAX=_env_int(env, "CODEX_WEB_RECENT_CWD_MAX", "256"),
        TRANSCRIPT_EXPORT_MAX_BYTES=_env_int(env, "CODEX_WEB_TRANSCRIPT_EXPORT_MAX_BYTES", str(50 * 1024 * 1024)),
    )
