from codoxear import server
from codoxear.session_manager_factories import session_manager_factory_caps


def test_factory_caps_capture_live_server_dependencies() -> None:
    """Factory construction exposes the server's concrete runtime dependencies."""
    caps = session_manager_factory_caps(server)

    assert caps.homes == {
        "codex": server.CODEX_HOME,
        "pi": server.PI_HOME,
        "cc": server.CC_HOME,
    }
    assert caps.tmux_session_name == server.TMUX_SESSION_NAME
    assert caps.recent_cwd_max == server.RECENT_CWD_MAX
    assert caps.prompt_prefix() == server._load_unattended_prompt(server.UNATTENDED_PROMPT_PATH)
    assert caps.run is server.subprocess.run
    assert caps.which_tmux is server.shutil.which
