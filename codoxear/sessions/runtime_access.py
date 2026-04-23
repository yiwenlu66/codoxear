from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from ..runtime import ServerRuntime, build_server_runtime


def manager_runtime(manager: Any) -> ServerRuntime:
    runtime = getattr(manager, "_runtime", None)
    if isinstance(runtime, ServerRuntime):
        return runtime

    module_name = getattr(getattr(manager, "__class__", None), "__module__", "")
    module = sys.modules.get(module_name)
    if not isinstance(module, ModuleType):
        raise RuntimeError("manager runtime is not initialized")

    event_hub = getattr(module, "EVENT_HUB", None)
    if event_hub is None:
        raise RuntimeError("manager runtime event hub is not initialized")

    runtime = build_server_runtime(module, manager=manager, event_hub=event_hub)
    manager._runtime = runtime
    return runtime
