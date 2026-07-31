from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
ReadBody = Callable[[Any], bytes]


@dataclass(frozen=True)
class HookRouteDeps:
    read_body: ReadBody
    json_response: JsonResponse


def handle_hook_post_route(handler: Any, *, path: str, deps: HookRouteDeps) -> bool:
    if path != "/api/hooks/notify":
        return False
    # Optional integration point. Current design does not rely on this.
    deps.read_body(handler)
    deps.json_response(handler, 200, {"ignored": True})
    return True
