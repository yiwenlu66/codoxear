from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from codoxear.server_config import (
    SERVER_CONFIG_EXPORT_NAMES,
    _validate_cookie_name,
    build_server_config,
    export_server_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_server_config_applies_dotenv_before_deriving_paths_without_overriding_env() -> None:
    with TemporaryDirectory() as td:
        cwd = Path(td)
        (cwd / ".env").write_text(
            "\n".join(
                [
                    "CODEX_WEB_URL_PREFIX=/phone",
                    "CODEX_WEB_COOKIE_SECURE=1",
                    "CODEX_WEB_TMUX_SESSION=mobile-agent",
                    "CODEX_HOME=/dotenv-codex",
                    "PI_HOME=/dotenv-pi",
                    "CLAUDE_CONFIG_DIR=/dotenv-cc",
                    "CODEX_WEB_ATTACH_MAX_BYTES=6",
                    "CODEX_WEB_DEFAULT_AGENT_BACKEND=pi",
                    "CODEX_WEB_PORT=12345",
                    "CODEX_WEB_UNATTENDED_SWEEP_SECONDS=7.5",
                    "CODEX_WEB_QUEUE_SWEEP_MAX_DRAINS=6",
                    "CODEX_WEB_QUEUE_SWEEP_MAX_ATTEMPTS=8",
                ]
            ),
            encoding="utf-8",
        )
        environ = {"PI_HOME": "/real-pi"}

        config = build_server_config(cwd=cwd, environ=environ)

    assert environ["CODEX_HOME"] == "/dotenv-codex"
    assert environ["PI_HOME"] == "/real-pi"
    assert environ["CLAUDE_CONFIG_DIR"] == "/dotenv-cc"
    assert config.CODEX_HOME == Path("/dotenv-codex")
    assert config.CODEX_SESSIONS_DIR == Path("/dotenv-codex/sessions")
    assert config.PI_HOME == Path("/real-pi")
    assert config.PI_SESSIONS_DIR == Path("/real-pi/agent/sessions")
    assert config.CC_HOME == Path("/dotenv-cc")
    assert config.CC_SESSIONS_DIR == Path("/dotenv-cc/projects")
    assert config.URL_PREFIX == "/phone"
    assert config.COOKIE_PATH == "/phone/"
    assert config.COOKIE_SECURE is True
    assert config.TMUX_SESSION_NAME == "mobile-agent"
    assert config.DEFAULT_AGENT_BACKEND == "pi"
    assert config.DEFAULT_PORT == 12345
    assert config.UNATTENDED_SWEEP_SECONDS == 7.5
    assert config.QUEUE_SWEEP_MAX_DRAINS == 6
    assert config.QUEUE_SWEEP_MAX_ATTEMPTS == 8
    assert config.SIDEBAR_PRIORITY_BUCKET_SECONDS == 30.0
    assert config.ATTACH_UPLOAD_MAX_BYTES == 6
    assert config.ATTACH_UPLOAD_BODY_MAX_BYTES == 65544
    assert config.UNATTENDED_PATH == config.APP_DIR / "unattended.json"
    assert config.UNATTENDED_PROMPT_PATH == config.APP_DIR / "unattended_prompt.txt"
    assert config.VIDEO_PREVIEW_DIR == config.APP_DIR / "video_previews"
    assert config.CC_SETTINGS_PATH == config.CC_HOME / "settings.json"


def test_export_server_config_populates_legacy_server_global_names() -> None:
    config = build_server_config(environ={})
    target = {"APP_DIR": "old"}

    export_server_config(target, config)

    assert target["_DOTENV"] == config.DOTENV_PATH
    assert target["APP_DIR"] == config.APP_DIR
    assert target["VIDEO_PREVIEW_DIR"] == config.VIDEO_PREVIEW_DIR
    assert target["UNATTENDED_PATH"] == config.UNATTENDED_PATH
    assert target["UNATTENDED_PROMPT_PATH"] == config.UNATTENDED_PROMPT_PATH
    assert target["CC_SETTINGS_PATH"] == config.CC_SETTINGS_PATH
    assert target["ATTACH_UPLOAD_MAX_BYTES"] == config.ATTACH_UPLOAD_MAX_BYTES
    assert set(SERVER_CONFIG_EXPORT_NAMES).issubset(target)


def test_queue_sweep_attempts_are_clamped_to_success_budget() -> None:
    config = build_server_config(environ={"CODEX_WEB_QUEUE_SWEEP_MAX_DRAINS": "5", "CODEX_WEB_QUEUE_SWEEP_MAX_ATTEMPTS": "2"})

    assert config.QUEUE_SWEEP_MAX_DRAINS == 5
    assert config.QUEUE_SWEEP_MAX_ATTEMPTS == 5


def test_cookie_name_default() -> None:
    config = build_server_config(environ={})
    assert config.COOKIE_NAME == "codoxear_auth"

    config = build_server_config(environ={"CODEX_WEB_COOKIE_NAME": ""})
    assert config.COOKIE_NAME == "codoxear_auth"


def test_cookie_name_custom() -> None:
    config = build_server_config(environ={"CODEX_WEB_COOKIE_NAME": "codoxear_preview_auth"})
    assert config.COOKIE_NAME == "codoxear_preview_auth"

    config = build_server_config(environ={"CODEX_WEB_COOKIE_NAME": "my_session"})
    assert config.COOKIE_NAME == "my_session"

    config = build_server_config(environ={"CODEX_WEB_COOKIE_NAME": "X"})
    assert config.COOKIE_NAME == "X"


def test_cookie_name_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_cookie_name("")


def test_cookie_name_rejects_separators() -> None:
    for bad in (";", " ", "(", ")", "<", ">", "@", ",", ":", "\\", '"', "/", "[", "]", "?", "=", "{", "}"):
        with pytest.raises(ValueError, match="invalid character"):
            _validate_cookie_name(f"codoxear{bad}auth")
    for bad in (" ", " codoxear", "codoxear ", "codoxear auth"):
        with pytest.raises(ValueError, match="invalid character"):
            build_server_config(environ={"CODEX_WEB_COOKIE_NAME": bad})


def test_cookie_name_rejects_control_chars() -> None:
    for cp in (0x00, 0x09, 0x0A, 0x1F, 0x7F):
        with pytest.raises(ValueError, match="invalid character"):
            _validate_cookie_name(f"bad{chr(cp)}cookie")


def test_queue_sweep_max_drains_config_is_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    config_source = (ROOT / "codoxear" / "server_config.py").read_text(encoding="utf-8")

    assert 'CODEX_WEB_QUEUE_SWEEP_MAX_DRAINS", "4"' in config_source
    assert 'CODEX_WEB_QUEUE_SWEEP_MAX_ATTEMPTS", "16"' in config_source
    assert "CODEX_WEB_QUEUE_SWEEP_MAX_DRAINS" in readme
    assert "maximum successful queued-prompt promotions per sweep" in readme
    assert "CODEX_WEB_QUEUE_SWEEP_MAX_ATTEMPTS" in readme
    assert "maximum queued sessions attempted per sweep" in readme
    assert "# CODEX_WEB_QUEUE_SWEEP_MAX_DRAINS=4" in env_example
    assert "# CODEX_WEB_QUEUE_SWEEP_MAX_ATTEMPTS=16" in env_example
