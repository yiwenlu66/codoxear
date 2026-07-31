from __future__ import annotations

from pathlib import Path
import copy
import threading
from typing import Any, Callable

from .launch_config import LaunchConfigPaths

LaunchDefaultsSignature = tuple[tuple[str, bool, int | None, int | None], ...]
LaunchDefaultsCache = tuple[LaunchDefaultsSignature, dict[str, Any]] | None


def path_signature(path: Path) -> tuple[str, bool, int | None, int | None]:
    try:
        stat_result = path.stat()
        return (str(path), True, int(stat_result.st_mtime_ns), int(stat_result.st_size))
    except (FileNotFoundError, OSError):
        return (str(path), False, None, None)


def launch_defaults_signature(paths: LaunchConfigPaths) -> LaunchDefaultsSignature:
    return tuple(path_signature(path) for path in vars(paths).values() if isinstance(path, Path))


def read_new_session_defaults_cached(
    *,
    paths_provider: Callable[[], LaunchConfigPaths],
    defaults_reader: Callable[..., dict[str, Any]],
    default_agent_backend: str,
    cache_lock: threading.Lock,
    get_cache: Callable[[], LaunchDefaultsCache],
    set_cache: Callable[[LaunchDefaultsCache], None],
) -> dict[str, Any]:
    paths = paths_provider()
    signature = launch_defaults_signature(paths)
    with cache_lock:
        cache = get_cache()
        if cache is not None and cache[0] == signature:
            return copy.deepcopy(cache[1])
    defaults = defaults_reader(paths, default_agent_backend=default_agent_backend)
    with cache_lock:
        set_cache((signature, copy.deepcopy(defaults)))
    return defaults


def launch_defaults_for_request(
    *,
    read_defaults: Callable[[], dict[str, Any]],
    fallback_defaults: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return read_defaults()
    except Exception:
        return fallback_defaults()
