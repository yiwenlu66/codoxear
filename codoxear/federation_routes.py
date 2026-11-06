"""HTTP routes for federation settings: /api/federation[ /token].

GET returns the effective config (tokens included — the UI is
password-protected and admins copy the hub token to nodes) plus live status.
POST replaces and applies the config at runtime; POST /api/federation/token
rotates the hub token. All routes use the normal UI auth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FederationRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: Callable[[Any, int, dict[str, Any]], None]
    read_json_body: Callable[..., dict[str, Any]]
    # Full GET payload: effective config + live status.
    status_payload: Callable[[], dict[str, Any]]
    # Persist + apply a replacement config; raises apply_error on failure.
    apply_settings: Callable[[dict[str, Any]], dict[str, Any]]
    # Rotate the hub token; returns {"token": ...}; raises apply_error.
    rotate_token: Callable[[], dict[str, Any]]
    apply_error: type[BaseException]


def _authorized(handler: Any, deps: FederationRouteDeps) -> bool:
    if deps.require_auth(handler):
        return True
    handler._unauthorized()
    return False


def handle_federation_get_route(handler: Any, *, path: str, deps: FederationRouteDeps) -> bool:
    if path != "/api/federation":
        return False
    if not _authorized(handler, deps):
        return True
    deps.json_response(handler, 200, {"ok": True, **deps.status_payload()})
    return True


def handle_federation_post_route(handler: Any, *, path: str, deps: FederationRouteDeps) -> bool:
    if path == "/api/federation/token":
        if not _authorized(handler, deps):
            return True
        try:
            result = deps.rotate_token()
        except deps.apply_error as exc:
            status = int(getattr(exc, "status", 500))
            deps.json_response(handler, status, {"ok": False, "error": str(exc), **deps.status_payload()})
            return True
        deps.json_response(handler, 200, {"ok": True, **result, **deps.status_payload()})
        return True
    if path != "/api/federation":
        return False
    if not _authorized(handler, deps):
        return True
    obj = deps.read_json_body(handler)
    try:
        deps.apply_settings(obj)
    except deps.apply_error as exc:
        # The file was already persisted; report the saved state plus the
        # apply error (e.g. 409 when the hub listen address is taken).
        status = int(getattr(exc, "status", 500))
        deps.json_response(handler, status, {"ok": False, "error": str(exc), **deps.status_payload()})
        return True
    deps.json_response(handler, 200, {"ok": True, **deps.status_payload()})
    return True
