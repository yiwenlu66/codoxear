"""Authenticated source-workspace discovery and explicit setup routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .source_workspace import SourceWorkspaceSetupError


SOURCE_WORKSPACE_PATH = "/api/source-workspace"


@dataclass(frozen=True)
class SourceWorkspaceRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: Callable[[Any, int, dict[str, Any]], None]
    read_body: Callable[[Any], bytes]
    discover: Callable[[], dict[str, Any]]
    setup: Callable[[], dict[str, Any]]


def handle_source_workspace_get_route(
    handler: Any,
    *,
    path: str,
    deps: SourceWorkspaceRouteDeps,
) -> bool:
    if path != SOURCE_WORKSPACE_PATH:
        return False
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    deps.json_response(handler, 200, deps.discover())
    return True


def handle_source_workspace_post_route(
    handler: Any,
    *,
    path: str,
    deps: SourceWorkspaceRouteDeps,
) -> bool:
    if path != SOURCE_WORKSPACE_PATH:
        return False
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    # This route takes no fields, but HTTP/1.1 requires consuming the body
    # before the connection can safely be reused for another request.
    deps.read_body(handler)
    try:
        result = deps.setup()
    except SourceWorkspaceSetupError as exc:
        deps.json_response(handler, exc.status, {"error": str(exc)})
        return True
    deps.json_response(handler, 200, result)
    return True
