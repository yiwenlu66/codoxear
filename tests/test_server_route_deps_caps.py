from codoxear import server
from codoxear.server_route_deps import ServerRouteDepsFactory


def test_route_dependency_factory_uses_its_supplied_server_config() -> None:
    """Route dependency values come from the explicit factory config."""
    factory = ServerRouteDepsFactory(server=server, config=server._SERVER_CONFIG)

    message_deps = factory.message_route_deps()
    session_deps = factory.session_route_deps()

    assert message_deps.transcript_export_max_bytes == server._SERVER_CONFIG.TRANSCRIPT_EXPORT_MAX_BYTES
    assert message_deps.transcript_search_max_line_bytes == server.TRANSCRIPT_SEARCH_MAX_LINE_BYTES
    assert session_deps.default_agent_backend == server._SERVER_CONFIG.DEFAULT_AGENT_BACKEND
    assert session_deps.tmux_session_name == server._SERVER_CONFIG.TMUX_SESSION_NAME
    assert message_deps.require_auth is server._require_auth
    assert session_deps.json_response is server._json_response
